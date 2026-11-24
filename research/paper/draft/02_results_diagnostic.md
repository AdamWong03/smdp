# Draft — Results 1: The diagnostic (provisional)

**Status:** pre-draft for Phase 4; numbers to be reconciled with the final gate.
**Figure:** Fig. 1; **Table:** Table 1.

---

## Pointwise accuracy does not order long-time fidelity

We first asked whether the conventional training objective—one-step displacement or velocity
error on held-out transitions—predicts the long-time behaviour of a force-free propagator. We
compared two models trained on the same SPC/Fw water data: our 250k relational propagator and the
published TrajCast water checkpoint. On 1,250 held-out transitions, the published checkpoint has a
mean absolute position error of \(4.75\times10^{-3}\) Å and a velocity error of
\(1.63\times10^{-3}\) Å fs\(^{-1}\). The 250k model is about fifteen times more accurate pointwise
(\(2.97\times10^{-4}\) Å and \(1.09\times10^{-4}\) Å fs\(^{-1}\)). If one-step accuracy controlled
long-time fidelity, the 250k model should be the better sampler.

It is not. Under the same 100 ps NVT protocol (300 K, CSVR, 5 fs), the published checkpoint
reproduces the canonical kinetic partition—translation \(2.505\) eV, rotation \(2.463\) eV,
internal \(2.444\) eV against the 2.5 ns reference means of \(2.459\), \(2.502\), and \(2.450\)
eV (the first 100 ps reference block is an outlier at 2.373/2.430/2.609 eV)—and gives
O–O, O–H, and H–H RDF RMSEs of 0.043, 0.118, and 0.096. The 250k model, despite its lower
one-step error, collapses the collective modes and overheats the internal mode: translation
\(0.538\) eV, rotation \(0.786\) eV, internal \(6.044\) eV. Its RDF RMSEs are 0.646, 1.312, and
0.249. The total kinetic energy is close to the target in both cases; the distribution across
modes is not. A pointwise-better model is a worse sampler, and the failure is invisible to the
training loss.

The inversion is not a peculiarity of one checkpoint. A 20k-step prefix of the same architecture
is even more pointwise accurate but becomes unstable in NVT after about 50 ps. A native TrajCast
architecture trained to only about 10% of its schedule has a pointwise error three times smaller
than the 250k model and fails in the opposite direction—translation \(3.256\) eV, rotation
\(3.105\) eV, internal \(1.041\) eV. The sign of the mode bias is architecture- and
training-dependent, but its presence is not.

## NVE energy drift is a red herring

A natural second hypothesis is that the failing models inject energy and heat, and that a
better energy integrator would fix the ensemble. We tested this directly with 25 ps NVE rollouts
from the same initial state. The published checkpoint heats from 302 K to 476 K; the 250k model
cools from 302 K to 271 K; the NVT-pair fine-tuned M0 model heats to 387 K. The ordering by NVE
drift is essentially the reverse of the ordering by NVT fidelity: the model that heats the most
passes the NVT screen, and the most NVE-stable model fails it. Energy conservation and canonical
sampling are different properties of the composed map. A force-free propagator can be a poor
energy integrator and a good thermostat, or vice versa; NVE drift cannot be used as a surrogate
for the stationary measure.

## The failure is a mode-resolved partition bias

The two diagnostics point to a common variable: the distribution of kinetic energy among
molecular modes. Figure 1c shows the mode kinetic energy relative to the reference for all models.
The failing models are not simply hot or cold. They are partitioned: the 250k family is
collectively cold and internally hot (translation 0.23, rotation 0.32, internal 2.32 of the
reference), the native/M0 family is collectively hot and internally cold (1.37, 1.28, 0.40), and
the published checkpoint is close to the reference on all three (1.06, 1.01, 0.94). The total
temperature is approximately 300 K in every case. A global thermostat holds the scalar
temperature while leaving the vector of mode temperatures biased.

This resolves the apparent paradox. The one-step loss is dominated by the conditional mean of the
displacement, which is insensitive to a slow redistribution of kinetic energy across modes. The
NVE trajectory measures the total energy budget, which can drift without changing the mode
partition in the same way. The NVT stationary measure is the object that sees the bias, and the
mode partition is its most informative low-dimensional projection.

## Stability is not correctness

The distinction matters because long rollouts are often used as the acceptance test. Figure 1d
plots the longest stable NVT run against the partition verdict. The phase-space v0 model is stable
for 2.75 ns and has a partition of 0.111/0.808/6.338 eV; the 250k and Current-Only models are
stable for 150 ps with the same bias. Stability without partition correctness is a long-lived
wrong answer, not a success. Conversely, short-horizon fixes can destabilize: the M0 endpoint
fine-tune passes on the val_04 start but blows up at 130 ps on an independent start, and the
5-step CSRV moment loss blows up at 4.3 ps. The acceptance criterion must be the stationary
measure, not the absence of a numerical crash.

These results define the design requirement for a remedy. The propagator must be paired with a
readout that controls the **mode-resolved** kinetic partition and the **multi-lag** dynamics, not
just the total temperature or a single-lag gain. The readout should not require retraining the
backbone, and it should be evaluated on the same long-time observables that expose the failure.
The next section introduces such a readout.
