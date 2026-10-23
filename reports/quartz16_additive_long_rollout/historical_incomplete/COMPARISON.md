# Previous / additive-mode iteration / BKS comparison

Previous: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1`
Current: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive`

Evaluation cadence, stopping thresholds, CSVR parameters and BKS are unchanged. A completed trajectory need not be physically accurate. Partial trajectories are not equivalent 15 ns evidence. Check recorded hardware before interpreting throughput differences.

| Run | Metric | Previous | Current | BKS | Current − previous |
|---|---|---:|---:|---:|---:|
| test | position_RMSE_A | 0.002095917 | 0.002806934 | unavailable | 0.0007110176 |
| test | velocity_RMSE_A_fs | 0.0003285606 | 0.0003871678 | unavailable | 5.860727e-05 |
| test | position_velocity_consistency_RMSE_A_fs | 0.000114423 | 0.0001006272 | unavailable | -1.379585e-05 |
| test | join_position_derivative_jump_RMS_A_fs | 0.0005700444 | 0.0004886633 | unavailable | -8.138109e-05 |
| test | two_step_position_RMSE_A | 0.00423301 | 0.005569565 | unavailable | 0.001336556 |
| nve_test_09 | production_available | 1 | 0 | unavailable | -1 |
| nve_test_10 | production_available | 1 | 0 | unavailable | -1 |
| nvt | production_available | 1 | 0 | unavailable | -1 |
