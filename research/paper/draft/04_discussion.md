# Draft — Discussion (provisional)

**Status:** pre-draft for Phase 4; final claims after the diffusion gate.
**Target:** ~700 words.

---

We have shown that the long-time failure of force-free molecular dynamics propagators is not
primarily a one-step accuracy problem. A published propagator with a fifteen-fold larger
pointwise error reproduces the canonical water ensemble, while a pointwise-better model develops
a large mode imbalance and fails the same NVT screen. The failure is also not an energy-drift
problem: the model that heats most strongly in NVE passes NVT, and the most NVE-stable model
fails. The discriminating object is the mode-resolved stationary measure of the composed
propagator plus thermostat. This diagnostic is, to our knowledge, the first systematic
demonstration of the pointwise–ensemble decoupling for force-free MD, and it changes what a
training or acceptance criterion should measure.

The remedy follows from the diagnosis. A frozen backbone supplies the structure—the displacement
that determines the position marginal—while a mode-resolved generalized-Langevin readout supplies
the velocity that enters the next backbone call. The readout fits a per-mode AR(\(p\)) memory
kernel to the reference velocity autocorrelation, injects noise with the
fluctuation–dissipation-consistent variance, and rescales each mode to its canonical kinetic
energy at every step. The partition is guaranteed by construction; the memory kernel matches the
multi-lag dynamics that a single gain cannot. The backbone is not retrained, and the readout is
fitted from a short reference NVT trajectory. On water, the result is a model that is stable on
two independent initial conditions, initial-condition independent in its partition, ACF, RDF,
and VDOS, and more accurate than the published checkpoint on the RDFs and VDOS in our analysis.

The relationship to prior work is complementary rather than competitive. TrajCast established
that force-free propagators can reproduce water physics [@trajcast]; our work explains when that
success is fragile and adds a readout that makes the mode partition a design constraint rather
than a lucky outcome. Structure-preserving action maps enforce symplectic or time-reversible
structure and improve energy conservation and equipartition [@bigi_action]; they target
mechanical structure, whereas the GLR targets the canonical NVT measure and the multi-lag
velocity statistics. The two could be combined: an action map would supply a better backbone,
and the GLR would enforce the mode partition on top of it. Learned velocity updates recover
stability with a black-box LSTM and periodic injection [@mamd]; the GLR is interpretable,
mode-resolved, and tied to the reference ACF, and it guarantees the partition rather than only
delaying divergence. GLE thermostats provide the same memory-kernel and colored-noise physics
[@ceriotti_prl; @ceriotti_jctc], and frequency-resolved thermostats can control selected modes
[@zpe_thermostat]; the difference is the object. A GLE thermostat acts on the forces and
velocities of an MD integrator. The GLR is a readout on a frozen force-free propagator, fitted
from the reference ACF, with an explicit per-mode variance guarantee and a diagnostic that
motivates it. Stochastic and generative propagators offer a different route to equilibrium
[@timewarp; @mdgen], at the cost of MCMC, an energy function, or a larger generative model; the
GLR is a lightweight post-hoc readout.

The diagnostic has implications beyond this method. Force-free propagators are usually selected
and reported by one-step MAE, sometimes with a short rollout score. Our results show that this
criterion can rank models in the wrong order. An ensemble-level, mode-resolved evaluation—kinetic
partition, RDF, VDOS, and diffusion—is necessary to detect the failure. The same lesson applies to
other learned dynamical maps: a low conditional-mean error does not imply a correct invariant
measure, and a stable rollout does not imply a correct one. We therefore recommend that
force-free MD papers report the mode partition and at least one long-time structural or
dynamical observable, not only one-step errors.

Several limitations define the boundary of the claim. First, the analytic guarantee covers the
**mean** mode kinetic partition and the velocity ACF, not the full joint position–velocity
distribution. The per-step rescale fixes the mode kinetic energy to its canonical mean and
therefore suppresses its equilibrium fluctuations (reference std about 0.24 eV versus about
0.002 eV in the readout). The marginal velocity distribution is nevertheless close to the
reference (component kurtosis 2.93/5.49/5.93 versus 3.04/5.62/5.57 for translation, rotation, and
internal motion), so the fixed-norm constraint does not distort the one-point velocity statistics.
The position marginal is learned by the backbone and validated empirically; a full canonical
guarantee, including the correct kinetic-energy fluctuations, would require a stochastic rescale
or an energy-based sampler such as MCMC. Second,
the readout requires a short reference NVT trajectory to fit the ACF. This is much cheaper than
retraining and is available for most systems of interest, but it is a data requirement, not a
free lunch. Third, we demonstrate one water model and one backbone family. The method is
backbone-agnostic in principle, but a second backbone and a second system are needed to establish
generality. Fourth, the internal ACF at 500 fs remains too persistent for AR(10); AR(20) reduces
the discrepancy but does not eliminate it. Fifth, the backbone uses a local 6 Å cutoff and no
long-range electrostatics; the GLR does not address long-range physics, and the water results
should not be read as a claim about large-scale dielectric or interfacial behaviour. Finally, the
current production runs use a single seed; seed and block error bars are needed before the
numerical claims are final.

The outlook is straightforward. The readout can be trained jointly with the backbone, rather than
fitted post hoc, which may improve the position marginal while retaining the partition guarantee.
The memory kernel can be made system- and temperature-adaptive, and the mode decomposition can be
extended to phonons or other collective coordinates. Combined with a structure-preserving
backbone and a long-range operator, the same diagnostic-and-readout strategy could make force-free
propagators reliable beyond water.
