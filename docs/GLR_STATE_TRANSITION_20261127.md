# GLR closed-loop state transition (2026-11-27)

**Scope:** current production code `code/h1h2/rollout_relational.py`,
`code/h1h2/relational_inference.py`, `code/h1h2/fd_readout.py`,
`code/h1h2/glr_readout.py`. No algorithm change.

## 1. Per-step transition

At step `t` the rollout holds unwrapped positions `q_t` and velocities `v_t`.

1. **Frozen backbone.** `step(model, q_t, v_t)` builds the periodic graph and calls the
   frozen R3-Opt model:
   - `dq_backbone = q_pred - q_t` (the model displacement),
   - `v_pred` (the model velocity).
2. **GLR readout.** `GLRReadout.forward(q_t, v_t)`:
   - decomposes `q_t, v_t` into translation / rotation / internal modes;
   - uses the AR history `H_{t-1}` of previous corrected mode velocities;
   - adds the fitted AR prediction and, in the stochastic variant, the FD noise;
   - returns `v_fd`.
3. **Blend and rescale.** `FDTransition.forward`:
   - `v_new = (1-λ) v_pred + λ v_fd`, `λ=0.1`;
   - per-mode variance rescale to `k_B T/2` (deterministic or stochastic target);
   - COM projection.
4. **State update.**
   - `q_{t+1} = q_t + dq_backbone` (the **current-step** displacement is the backbone
     displacement, not `h v_new`);
   - `v_{t+1} = v_new`;
   - `H_t = H_{t-1} ∪ {modes(q_t, v_new)}` (`GLRReadout.update`).
5. **Next step.** The next `step(model, q_{t+1}, v_{t+1})` sends **`v_new` back into the
   frozen backbone**. The corrected velocity therefore changes the next backbone
   displacement and all subsequent positions.

No external CSVR is applied in the production GLR run (`--ensemble nve`; the stored
`thermostat_alpha` is 1.0). The GLR per-mode rescale is the thermostat. In raw/CSVR runs the
external CSVR scales `v` after the model, and the next backbone sees the scaled `v`; the same
feedback path exists.

## 2. Diagram

```text
                 q_t, v_t
                    |
                    v
        +---------------------------+
        | frozen R3-Opt backbone    |
        |  (q_t, v_t) -> dq_b, v_pred|
        +---------------------------+
             |                |
       dq_b  |                | v_pred
             |                v
             |     +---------------------------+
             |     | GLR readout               |
             |     | modes(q_t,v_t), H_{t-1}   |
             |     | AR + noise -> v_fd        |
             |     +---------------------------+
             |                |
             |                v
             |     +---------------------------+
             |     | blend λ=0.1 + rescale     |
             |     | v_new = .9 v_pred+.1 v_fd |
             |     +---------------------------+
             |                |
             v                v
        q_{t+1}=q_t+dq_b   v_{t+1}=v_new
             |                |
             +-------+--------+
                     |
                     v
        next backbone sees q_{t+1}, v_{t+1}
        H_t = H_{t-1} + modes(q_t, v_new)
```

The loop is **closed**: `v_new` is not a post-hoc trajectory correction. The only local
non-equality is `q_{t+1}-q_t ≠ h v_new`; the correct statement is that the current displacement
comes from the backbone while the corrected velocity is the next state and drives future
backbone calls.

## 3. One-step feedback probe

Production checkpoint, production GLR coefficients, deterministic readout, same `FDTransition`;
one step from a matched state. `v_new` is fed to a second backbone call at the same `q_t`; the
next displacement is compared with the `v_pred` call.

| state | correction / v_pred | next displacement change / dq0 | next v_pred change / v_pred | correction·v_pred cosine |
|---|---:|---:|---:|---:|
| val_04 frame 0 | 0.218 | **0.632** | **1.131** | -0.032 |
| GLR 2.5 ns frame 20,000 (100 ps) | 0.209 | **0.617** | **1.143** | -0.041 |

The correction is ~21% of `v_pred`, but the frozen backbone's next displacement changes by
~62% of `dq0` and its next velocity changes by ~113% of `v_pred`. The feedback is strong and
nearly orthogonal to the current velocity. This is why the closed loop cannot be described as a
post-hoc correction.

## 4. Feedback intervention (running)

Matched 100 ps deterministic runs, same checkpoint / val_04 frame 0 / 5 fs / GLR coefficients:

| mode | feedback velocity | purpose |
|---|---|---|
| `full` | `v_new` | production closed loop |
| `no_feedback` | `v_pred` | removes backbone feedback; readout history still has `v_new` |
| `all_no_feedback` | `v_pred` + history reset to `v_pred` | removes both backbone and readout-history feedback |
| `thermostat_only` | rescaled `v_pred` | matches the KE thermostat without AR/blend |

The compare isolates the feedback from the thermostat and from RNG: the deterministic readout
has no noise; all runs share the initial state and checkpoint. Results will be appended to
`experiments/feedback_*_100ps_20261127/` and the audit.
