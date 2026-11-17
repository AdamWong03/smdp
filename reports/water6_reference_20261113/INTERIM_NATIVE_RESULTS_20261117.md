# Interim native TrajCast NVE result, 2026-11-17

The immutable epoch-75 snapshot completed all 4,000 official Forecast steps
(20 ps). Requested motion/RDF/VDOS/diffusion analyses completed. This is an
interim validation result, not the final selected model, final test evaluation,
energy-verified analysis, or NVT reproduction. No parameters were changed from
the result and training has not been stopped.

All three models start at the identical val_04/frame 0 (temperature 275.243 K).
There is no thermostat and no reference force call during propagation.

| Measurement | DSRT 4.5 Å final best | DSRT 6 Å final best | TrajCast epoch75 | Reference MD |
|---|---:|---:|---:|---:|
| 0–20 ps mean temperature, K | 369.843 | 296.553 | 587.586 | 274.994 |
| Temperature at 20 ps, K | 388.715 | 245.768 | 776.180 | — |
| Mean molecular translation KE, eV | 1.40235 | 1.36561 | 3.79216 | 2.21525 |
| Mean molecular rotation KE, eV | 1.11604 | 1.53773 | 4.09075 | 2.23611 |
| Mean molecular internal KE, eV | 6.61254 | 4.41816 | 6.62379 | 2.33788 |

No inference of equilibrium quality should be made from these nonstationary
means. The 6 Å DSRT initially warms then cools: its near-300 K overall mean
conceals a final temperature below reference. TrajCast epoch75 warms strongly,
with all molecular KE components excessive. Its lower endpoint validation
error did not yield accurate 20 ps unthermostatted dynamics in this diagnosis.

This comparison uses different training stages and optimizer/precision/budget
settings. It does not establish that the native architecture is inferior or
that its final selected model will behave the same. The official paper's long
trajectory results use CSVR; this NVE diagnosis does not test those NVT claims.
The earlier inference that richer native geometry would necessarily solve
autonomous physics is unsupported. Short-term accuracy and autonomous dynamics
must continue to be evaluated separately.

## Continuing training

102 epochs completed, epoch 103 in progress at inspection. Validation-best is
epoch 101 / 252,500 updates, position RMSE 0.000156216 Å and velocity RMSE
0.0000710063 Å/fs. It has not undergone independent final test evaluation or
autonomous evaluation. Do not label the epoch75 result as its physical result.

## Implementation checks and limits

The snapshot's first Forecast position update matched its own rescaled
readout to 1.78e-15 Å and velocity to 1.56e-17 Å/fs. Native graph rebuilding,
normalization and momentum layer were used unchanged. This checks adapter
wiring; it does not prove physical fidelity.

No verified total energy drift is reported: the reference energy acceptance
issue remains unresolved and energy was not requested in this interim run.
Numerical completion (`stable=true` in the summary) means no nonfinite state
or execution failure, not physical stability. No new architecture or loss is
authorized by this result. Raw trajectory, snapshot and full logs stay under
`experiments/water6_reference_20261113/trajcast_native/interim_epoch75/`.
