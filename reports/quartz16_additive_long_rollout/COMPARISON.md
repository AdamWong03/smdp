# Previous / additive-mode iteration / BKS comparison

Previous: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1`
Current: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive`

Evaluation cadence, stopping thresholds, CSVR parameters and BKS are unchanged. A completed trajectory need not be physically accurate. Partial trajectories are not equivalent 15 ns evidence. Check recorded hardware before interpreting throughput differences.

Trajectory generation and physical analysis are separate states:

```json
{
  "nve_test_09": {
    "trajectory_completed": true,
    "production_ps": 100.0,
    "physical_analysis_completed": true,
    "analysis_failure_reason": {}
  },
  "nve_test_10": {
    "trajectory_completed": true,
    "production_ps": 100.0,
    "physical_analysis_completed": true,
    "analysis_failure_reason": {}
  },
  "nvt": {
    "trajectory_completed": true,
    "production_ps": 15000.0,
    "physical_analysis_completed": true,
    "analysis_failure_reason": {}
  }
}
```

| Run | Metric | Previous | Current | BKS | Current − previous |
|---|---|---:|---:|---:|---:|
| test | position_RMSE_A | 0.002095917 | 0.002806934 | unavailable | 0.0007110176 |
| test | velocity_RMSE_A_fs | 0.0003285606 | 0.0003871678 | unavailable | 5.860727e-05 |
| test | position_velocity_consistency_RMSE_A_fs | 0.000114423 | 0.0001006272 | unavailable | -1.379585e-05 |
| test | join_position_derivative_jump_RMS_A_fs | 0.0005700444 | 0.0004886633 | unavailable | -8.138109e-05 |
| test | two_step_position_RMSE_A | 0.00423301 | 0.005569565 | unavailable | 0.001336556 |
| nve_test_09 | production_ps | 100 | 100 | unavailable | 0 |
| nve_test_09 | ns_day_end_to_end | 162.226 | 109.008 | unavailable | -53.21799 |
| nve_test_09 | ms_per_step | 8.521446 | 12.68164 | unavailable | 4.160192 |
| nve_test_09 | peak_gpu_bytes | 1.242516e+08 | 1.244595e+08 | unavailable | 207872 |
| nve_test_09 | temperature_K_mean | 700.5482 | 254.284 | 287.9095 | -446.2642 |
| nve_test_09 | temperature_K_std | 96.99214 | 44.55164 | 12.80922 | -52.4405 |
| nve_test_09 | kinetic_energy_eV_mean | 14.57901 | 5.291868 | 5.991644 | -9.287142 |
| nve_test_09 | kinetic_energy_eV_std | 2.01849 | 0.927158 | 0.2665708 | -1.091332 |
| nve_test_09 | potential_energy_eV_mean | -3127.569 | -3136.791 | -3136.012 | -9.222167 |
| nve_test_09 | potential_energy_eV_std | 1.967873 | 0.8801856 | 0.2678619 | -1.087687 |
| nve_test_09 | energy_change_eV | 17.42939 | 0.1394124 | 0.00478189 | -17.28998 |
| nve_test_09 | total_energy_drift_meV_atom_ps | 0.1389063 | 0.1125855 | 0.000576153 | -0.02632077 |
| nve_test_09 | minimum_distance_A | 1.401021 | 1.489619 | unavailable | 0.08859789 |
| nve_test_09 | species_KE_eV_Z8 | 10.19382 | 3.563363 | 3.995259 | -6.630459 |
| nve_test_09 | species_KE_eV_Z14 | 4.385188 | 1.728505 | 1.996385 | -2.656682 |
| nve_test_09 | macro_16fs_VDOS_overlap_Z8 | 0.734408 | 0.6883082 | 1 | -0.04609976 |
| nve_test_09 | macro_16fs_VDOS_overlap_Z14 | 0.7947465 | 0.7149275 | 1 | -0.07981908 |
| nve_test_09 | queried_1fs_VDOS_overlap_Z8 | 0.7331088 | 0.7015094 | 1 | -0.03159944 |
| nve_test_09 | queried_1fs_VDOS_overlap_Z14 | 0.7859662 | 0.7146763 | 1 | -0.07128989 |
| nve_test_09 | RDF_RMSE_14-8 | 1.308631 | 0.6863299 | 0 | -0.6223008 |
| nve_test_09 | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nve_test_09 | RDF_14-8_peak_height | 22.73503 | 43.10842 | 35.36422 | 20.3734 |
| nve_test_09 | RDF_RMSE_8-8 | 0.3948633 | 0.07114615 | 0 | -0.3237171 |
| nve_test_09 | RDF_8-8_peak_position_A | 2.624728 | 2.624728 | 2.600975 | 0 |
| nve_test_09 | RDF_8-8_peak_height | 6.69243 | 8.995441 | 9.247842 | 2.303012 |
| nve_test_10 | production_ps | 100 | 100 | unavailable | 0 |
| nve_test_10 | ns_day_end_to_end | 165.6799 | 116.309 | unavailable | -49.37089 |
| nve_test_10 | ms_per_step | 8.3438 | 11.88558 | unavailable | 3.541779 |
| nve_test_10 | peak_gpu_bytes | 1.242516e+08 | 1.244595e+08 | unavailable | 207872 |
| nve_test_10 | temperature_K_mean | 707.7279 | 274.8235 | 302.7117 | -432.9044 |
| nve_test_10 | temperature_K_std | 98.2023 | 63.22961 | 13.54457 | -34.97269 |
| nve_test_10 | kinetic_energy_eV_mean | 14.72843 | 5.719313 | 6.299691 | -9.009113 |
| nve_test_10 | kinetic_energy_eV_std | 2.043674 | 1.315863 | 0.2818741 | -0.7278118 |
| nve_test_10 | potential_energy_eV_mean | -3127.374 | -3136.376 | -3135.705 | -9.001793 |
| nve_test_10 | potential_energy_eV_std | 2.021347 | 1.284291 | 0.2832843 | -0.7370557 |
| nve_test_10 | energy_change_eV | 19.11145 | -2.24998 | 0.003283002 | -21.36143 |
| nve_test_10 | total_energy_drift_meV_atom_ps | 0.1447016 | 0.2157748 | -0.0004707605 | 0.07107326 |
| nve_test_10 | minimum_distance_A | 1.41664 | 1.500829 | unavailable | 0.08418941 |
| nve_test_10 | species_KE_eV_Z8 | 10.36138 | 3.960274 | 4.217151 | -6.401106 |
| nve_test_10 | species_KE_eV_Z14 | 4.367045 | 1.759039 | 2.08254 | -2.608007 |
| nve_test_10 | macro_16fs_VDOS_overlap_Z8 | 0.7301604 | 0.6608274 | 1 | -0.069333 |
| nve_test_10 | macro_16fs_VDOS_overlap_Z14 | 0.7958644 | 0.7190893 | 1 | -0.07677506 |
| nve_test_10 | queried_1fs_VDOS_overlap_Z8 | 0.7348608 | 0.6752904 | 1 | -0.05957041 |
| nve_test_10 | queried_1fs_VDOS_overlap_Z14 | 0.7921053 | 0.7162824 | 1 | -0.07582289 |
| nve_test_10 | RDF_RMSE_14-8 | 1.213482 | 0.700744 | 0 | -0.5127376 |
| nve_test_10 | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nve_test_10 | RDF_14-8_peak_height | 23.02497 | 42.25678 | 34.6475 | 19.23181 |
| nve_test_10 | RDF_RMSE_8-8 | 0.3872451 | 0.07538397 | 0 | -0.3118611 |
| nve_test_10 | RDF_8-8_peak_position_A | 2.624728 | 2.624728 | 2.600975 | 0 |
| nve_test_10 | RDF_8-8_peak_height | 6.654587 | 8.912908 | 9.071304 | 2.258321 |
| nvt | production_ps | 15000 | 15000 | unavailable | 0 |
| nvt | ns_day_end_to_end | 163.2522 | 104.9099 | unavailable | -58.34232 |
| nvt | ms_per_step | 8.467878 | 13.17702 | unavailable | 4.709142 |
| nvt | peak_gpu_bytes | 1.242501e+08 | 1.244611e+08 | unavailable | 210944 |
| nvt | temperature_K_mean | 361.3444 | 280.7478 | 299.184 | -80.59658 |
| nvt | temperature_K_std | 28.76607 | 19.71181 | 18.65835 | -9.054261 |
| nvt | kinetic_energy_eV_mean | 7.519887 | 5.842603 | 6.226276 | -1.677284 |
| nvt | kinetic_energy_eV_std | 0.5986466 | 0.4102197 | 0.3882962 | -0.188427 |
| nvt | potential_energy_eV_mean | -3134.682 | -3136.164 | -3135.773 | -1.481436 |
| nvt | potential_energy_eV_std | 0.5775837 | 0.403189 | 0.4010646 | -0.1743947 |
| nvt | energy_change_eV | -0.8342773 | 0.1137354 | -1.218369 | 0.9480127 |
| nvt | total_energy_drift_meV_atom_ps | -2.124982e-05 | -3.299266e-06 | -8.948068e-06 | 1.795056e-05 |
| nvt | minimum_distance_A | 1.470124 | 1.478557 | unavailable | 0.008432746 |
| nvt | species_KE_eV_Z8 | 6.05361 | 3.968541 | 4.150435 | -2.085068 |
| nvt | species_KE_eV_Z14 | 1.466278 | 1.874062 | 2.075842 | 0.4077843 |
| nvt | macro_16fs_VDOS_overlap_Z8 | 0.4496718 | 0.8109382 | 1 | 0.3612664 |
| nvt | macro_16fs_VDOS_overlap_Z14 | 0.6281193 | 0.831344 | 1 | 0.2032247 |
| nvt | queried_1fs_VDOS_overlap_Z8 | 0.4577341 | 0.8188459 | 1 | 0.3611118 |
| nvt | queried_1fs_VDOS_overlap_Z14 | 0.6196627 | 0.8339806 | 1 | 0.2143178 |
| nvt | RDF_RMSE_14-8 | 0.2401528 | 0.570315 | 0 | 0.3301622 |
| nvt | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nvt | RDF_14-8_peak_height | 36.92971 | 41.0664 | 34.76711 | 4.136686 |
| nvt | RDF_RMSE_8-8 | 0.3096488 | 0.05685032 | 0 | -0.2527985 |
| nvt | RDF_8-8_peak_position_A | 2.648481 | 2.600975 | 2.600975 | -0.04750639 |
| nvt | RDF_8-8_peak_height | 7.931828 | 8.757685 | 9.167464 | 0.825857 |
