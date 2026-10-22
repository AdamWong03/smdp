# Quartz 16 fs: observed results

Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive`

Model: fixed architecture with learned motion tokens, mode-conditioned recurrent/temporal routing, mid-evolution cross-mode exchange, equivariant segment decoding.
All comparisons below use BKS reference MD. No claim of named physical modes follows from routing weights.

## Training

Completed epoch records: 136. Training wall time: 5.252 h.

## Independent endpoint test

```json
{
  "checkpoint": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive/best_rollout.pt",
  "test_samples": 1000,
  "endpoint": {
    "time_fs": 16,
    "position_MAE_A": 0.0022014024024829266,
    "position_RMSE_A": 0.0028069344272189005,
    "velocity_MAE_A_fs": 0.0002991476475726813,
    "velocity_RMSE_A_fs": 0.0003871678234989092
  },
  "normalized_endpoint_loss": 0.008581168547484056,
  "position_velocity_consistency_RMSE_A_fs": 0.00010062719949219023,
  "reference_finite_difference_RMSE_A_fs": 8.725477203009129e-06,
  "join_position_derivative_jump_RMS_A_fs": 0.0004886632994872442,
  "join_state_discontinuity": 0.0,
  "two_step_position_RMSE_A": 0.005569565278678313,
  "reference_join_derivative_jump_RMS_A_fs": 0.0004941729594609555,
  "multi_step": {
    "2": {
      "time_fs": 32,
      "samples": 991,
      "position_RMSE_A": 0.005569496883717362,
      "velocity_RMSE_A_fs": 0.00046195440175602584
    },
    "4": {
      "time_fs": 64,
      "samples": 991,
      "position_RMSE_A": 0.010254738325191878,
      "velocity_RMSE_A_fs": 0.0006289296667422746
    },
    "8": {
      "time_fs": 128,
      "samples": 991,
      "position_RMSE_A": 0.019266563498981957,
      "velocity_RMSE_A_fs": 0.0008887455798372462
    },
    "16": {
      "time_fs": 256,
      "samples": 991,
      "position_RMSE_A": 0.03593630201509356,
      "velocity_RMSE_A_fs": 0.001349019669200903
    }
  },
  "mode_statistics": {
    "scalar_allocation": [
      0.20430104494094847,
      0.24354276341199874,
      0.24407702267169953,
      0.30807918071746826
    ],
    "vector_allocation": [
      0.08011368241906167,
      0.33251252508163454,
      0.21801871716976165,
      0.3693550822734833
    ],
    "recurrent_route_by_mode": [
      0.9474980506896973,
      0.9999426696300506,
      0.736204607963562,
      0.6906679615974426
    ],
    "cross_mode_attention": [
      [
        0.0,
        0.32653590095043183,
        0.0027725804718211292,
        0.6706915247440338
      ],
      [
        0.00019062258693156763,
        0.0,
        0.6624347939491272,
        0.3373745856285095
      ],
      [
        1.3659281677973922e-05,
        5.823072444763966e-05,
        0.0,
        0.9999281134605408
      ],
      [
        2.0134558661084157e-05,
        0.333310960650444,
        0.6666689097881318,
        0.0
      ]
    ],
    "mode_delta_q_rms": [
      0.008962896142154932,
      0.015340653747320175,
      0.013631695073097944,
      0.027945060230791567
    ],
    "mode_delta_q_rms_share": [
      0.13607161676883697,
      0.2328813772201538,
      0.2069404634833336,
      0.42410653877258303
    ],
    "mode_delta_q_signed_projection": [
      0.09836288821697235,
      0.1859840551018715,
      0.18246646040678025,
      0.5331865935325623
    ],
    "mode_delta_v_rms": [
      0.0009955811973195523,
      0.0019103168356232345,
      0.0011927563692443073,
      0.0038067240891978145
    ],
    "mode_delta_v_rms_share": [
      0.12597550264000892,
      0.24166092777252196,
      0.1509008665084839,
      0.48146270513534545
    ],
    "mode_delta_v_signed_projection": [
      0.06611055496335029,
      0.17319196420907973,
      0.11869033697247505,
      0.642007138967514
    ]
  }
}
```

## Autonomous propagation

| Run | Production ps | Failure | ns/day (including I/O) |
|---|---:|---|---:|
| nve_test_09 | 100 | none recorded | 109 |
| nve_test_10 | 100 | none recorded | 116.3 |
| nvt | 15000 | none recorded | 104.9 |

## Observed mechanism and cost limitations

Additive mode readout; no competitive fusion weights. Report actual per-mode displacement/velocity RMS and signed projections from the selected checkpoint test. These attributions are not named physical modes.
Soft routing evaluates both recurrent and temporal operators. Inference cost includes all 16 time queries; a learned route does not save expert computation. No matched-hardware BKS speedup is claimed without a measured baseline.

## Interpretation boundaries

Only completed production frames are production evidence. Failed thermalization statistics are diagnostics, not 15 ns NVT results.
The stable field only means no numerical termination threshold was triggered. It is not evidence of physical fidelity or energy conservation.
16 fs spectra compare identically downsampled outputs. Queried 1 fs spectra compare dense velocities separately and include endpoint thermostat kicks; output cadence alone does not establish high-frequency fidelity.
This experiment stops after these evaluations; there is no automatic architecture change or second training run.
