# Interim native validation diagnosis, 2026-11-16

The user asked whether TrajCast can already be assessed without waiting for
the end of its long native training schedule. The current validation-selected
epoch-75 checkpoint (187,500 updates) was copied to an immutable snapshot:

`experiments/water6_reference_20261113/trajcast_native/interim_epoch75/snapshot.pt`

Training is not stopped or modified. This is a provisional validation diagnosis,
not a final independent test or a new checkpoint-selection criterion. Do not
tune the model, simulation or training from its outcome. Final test selection
continues to follow the frozen validation endpoint objective.

Run the same val_04/frame 0, 20 ps NVE used by the two DSRT checkpoints.
`water_reference.diagnose_native_nve` loads the native FP64 model strictly and
constructs the official `Forecast` object. Every step calls its unchanged
`Forecast._make_timestep` (native normalization, momentum layer, coordinate
wrap and graph rebuild). There is no thermostat or reference force call.
Only HDF5 recording and continuous-image reconstruction for offline analysis
are added. A real-input check compares the first Forecast step with the model's
own rescaled output before propagation. No reduced-data training is performed.

Offline tasks are molecular motion/geometry, RDF, VACF/VDOS and diffusion,
using existing definitions and exactly the same reference time window. The
previous energy-reference verification failure remains unresolved; energy
metrics are not silently treated as verified in this diagnosis.

Launch (already running on shared L40S GPU 2; do not duplicate):

```bash
CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900 \
  bash scripts/run_native_interim_nve.sh
```

Logs: `trajcast_native/interim_epoch75/rollout.log`, `analysis.log`,
`pipeline.log` under the server experiment directory. Results are pending.

At snapshot time: 75 training epochs completed, epoch 76 running; best
validation q/v RMSE 0.000177392 Å / 0.0000773878 Å/fs. DSRT 6 Å final validation
q/v RMSE is 0.000411116 Å / 0.000149922 Å/fs. This establishes an endpoint
accuracy difference but no autonomous-physics conclusion.

Recent native epochs take approximately 3,090 seconds each. At that speed,
100 further stale epochs take about 3.6 days, conditional on no new best;
running the maximum remaining 1,425 epochs takes about 51 days. These are
conditional projections under shared load, not predicted completion dates.
No early-stop rule, learning rate or training budget has been changed.
