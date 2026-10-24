# Previous / additive-mode iteration / BKS comparison

Previous: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive`
Current: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition`

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
| test | position_RMSE_A | 0.002806934 | 0.00249977 | unavailable | -0.0003071647 |
| test | velocity_RMSE_A_fs | 0.0003871678 | 0.0003425479 | unavailable | -4.461988e-05 |
| test | position_velocity_consistency_RMSE_A_fs | 0.0001006272 | 9.084611e-05 | unavailable | -9.781087e-06 |
| test | join_position_derivative_jump_RMS_A_fs | 0.0004886633 | 0.0004751208 | unavailable | -1.354252e-05 |
| test | two_step_position_RMSE_A | 0.005569565 | 0.005025742 | unavailable | -0.0005438235 |
| nve_test_09 | production_ps | 100 | 100 | unavailable | 0 |
| nve_test_09 | ns_day_end_to_end | 109.008 | 93.38516 | unavailable | -15.62284 |
| nve_test_09 | ms_per_step | 12.68164 | 14.80321 | unavailable | 2.121571 |
| nve_test_09 | peak_gpu_bytes | 1.244595e+08 | 1.100723e+09 | unavailable | 9.762632e+08 |
| nve_test_09 | temperature_K_mean | 254.284 | 501.027 | 287.9095 | 246.743 |
| nve_test_09 | temperature_K_std | 44.55164 | 71.85561 | 12.80922 | 27.30397 |
| nve_test_09 | kinetic_energy_eV_mean | 5.291868 | 10.4268 | 5.991644 | 5.134933 |
| nve_test_09 | kinetic_energy_eV_std | 0.927158 | 1.495377 | 0.2665708 | 0.5682191 |
| nve_test_09 | potential_energy_eV_mean | -3136.791 | -3131.553 | -3136.012 | 5.237981 |
| nve_test_09 | potential_energy_eV_std | 0.8801856 | 1.487002 | 0.2678619 | 0.6068164 |
| nve_test_09 | energy_change_eV | 0.1394124 | 10.71203 | 0.00478189 | 10.57262 |
| nve_test_09 | total_energy_drift_meV_atom_ps | 0.1125855 | 0.07756059 | 0.000576153 | -0.03502492 |
| nve_test_09 | minimum_distance_A | 1.489619 | 1.431252 | unavailable | -0.05836678 |
| nve_test_09 | species_KE_eV_Z8 | 3.563363 | 7.636763 | 3.995259 | 4.073401 |
| nve_test_09 | species_KE_eV_Z14 | 1.728505 | 2.790038 | 1.996385 | 1.061533 |
| nve_test_09 | macro_16fs_VDOS_overlap_Z8 | 0.6883082 | 0.7100392 | 1 | 0.02173097 |
| nve_test_09 | macro_16fs_VDOS_overlap_Z14 | 0.7149275 | 0.7964666 | 1 | 0.0815391 |
| nve_test_09 | queried_1fs_VDOS_overlap_Z8 | 0.7015094 | 0.7078621 | 1 | 0.006352771 |
| nve_test_09 | queried_1fs_VDOS_overlap_Z14 | 0.7146763 | 0.7862234 | 1 | 0.07154718 |
| nve_test_09 | RDF_RMSE_14-8 | 0.6863299 | 0.908497 | 0 | 0.2221671 |
| nve_test_09 | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nve_test_09 | RDF_14-8_peak_height | 43.10842 | 26.37317 | 35.36422 | -16.73525 |
| nve_test_09 | RDF_RMSE_8-8 | 0.07114615 | 0.3038539 | 0 | 0.2327077 |
| nve_test_09 | RDF_8-8_peak_position_A | 2.624728 | 2.624728 | 2.600975 | 0 |
| nve_test_09 | RDF_8-8_peak_height | 8.995441 | 7.39847 | 9.247842 | -1.596971 |
| nve_test_10 | production_ps | 100 | 100 | unavailable | 0 |
| nve_test_10 | ns_day_end_to_end | 116.309 | 92.19672 | unavailable | -24.1123 |
| nve_test_10 | ms_per_step | 11.88558 | 14.99403 | unavailable | 3.108447 |
| nve_test_10 | peak_gpu_bytes | 1.244595e+08 | 1.099412e+09 | unavailable | 9.749524e+08 |
| nve_test_10 | temperature_K_mean | 274.8235 | 506.1892 | 302.7117 | 231.3657 |
| nve_test_10 | temperature_K_std | 63.22961 | 75.04243 | 13.54457 | 11.81282 |
| nve_test_10 | kinetic_energy_eV_mean | 5.719313 | 10.53423 | 6.299691 | 4.814919 |
| nve_test_10 | kinetic_energy_eV_std | 1.315863 | 1.561697 | 0.2818741 | 0.2458349 |
| nve_test_10 | potential_energy_eV_mean | -3136.376 | -3131.416 | -3135.705 | 4.960161 |
| nve_test_10 | potential_energy_eV_std | 1.284291 | 1.547572 | 0.2832843 | 0.263281 |
| nve_test_10 | energy_change_eV | -2.24998 | 8.570923 | 0.003283002 | 10.8209 |
| nve_test_10 | total_energy_drift_meV_atom_ps | 0.2157748 | 0.1020809 | -0.0004707605 | -0.1136939 |
| nve_test_10 | minimum_distance_A | 1.500829 | 1.450849 | unavailable | -0.04998052 |
| nve_test_10 | species_KE_eV_Z8 | 3.960274 | 7.743993 | 4.217151 | 3.783719 |
| nve_test_10 | species_KE_eV_Z14 | 1.759039 | 2.790238 | 2.08254 | 1.0312 |
| nve_test_10 | macro_16fs_VDOS_overlap_Z8 | 0.6608274 | 0.7094248 | 1 | 0.0485974 |
| nve_test_10 | macro_16fs_VDOS_overlap_Z14 | 0.7190893 | 0.7972499 | 1 | 0.07816055 |
| nve_test_10 | queried_1fs_VDOS_overlap_Z8 | 0.6752904 | 0.7119126 | 1 | 0.03662216 |
| nve_test_10 | queried_1fs_VDOS_overlap_Z14 | 0.7162824 | 0.7893666 | 1 | 0.07308419 |
| nve_test_10 | RDF_RMSE_14-8 | 0.700744 | 0.8419543 | 0 | 0.1412103 |
| nve_test_10 | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nve_test_10 | RDF_14-8_peak_height | 42.25678 | 26.18944 | 34.6475 | -16.06734 |
| nve_test_10 | RDF_RMSE_8-8 | 0.07538397 | 0.2833741 | 0 | 0.2079901 |
| nve_test_10 | RDF_8-8_peak_position_A | 2.624728 | 2.624728 | 2.600975 | 0 |
| nve_test_10 | RDF_8-8_peak_height | 8.912908 | 7.331074 | 9.071304 | -1.581834 |
| nvt | production_ps | 15000 | 15000 | unavailable | 0 |
| nvt | ns_day_end_to_end | 104.9099 | 29.86337 | unavailable | -75.04654 |
| nvt | ms_per_step | 13.17702 | 46.29082 | unavailable | 33.1138 |
| nvt | peak_gpu_bytes | 1.244611e+08 | 1.101076e+09 | unavailable | 9.766149e+08 |
| nvt | temperature_K_mean | 280.7478 | 319.4667 | 299.184 | 38.71888 |
| nvt | temperature_K_std | 19.71181 | 26.00094 | 18.65835 | 6.289127 |
| nvt | kinetic_energy_eV_mean | 5.842603 | 6.648376 | 6.226276 | 0.8057731 |
| nvt | kinetic_energy_eV_std | 0.4102197 | 0.5411018 | 0.3882962 | 0.1308822 |
| nvt | potential_energy_eV_mean | -3136.164 | -3135.328 | -3135.773 | 0.8355237 |
| nvt | potential_energy_eV_std | 0.403189 | 0.5288545 | 0.4010646 | 0.1256655 |
| nvt | energy_change_eV | 0.1137354 | -0.2576006 | -1.218369 | -0.3713361 |
| nvt | total_energy_drift_meV_atom_ps | -3.299266e-06 | -6.191673e-06 | -8.948068e-06 | -2.892408e-06 |
| nvt | minimum_distance_A | 1.478557 | 1.457161 | unavailable | -0.02139568 |
| nvt | species_KE_eV_Z8 | 3.968541 | 5.023429 | 4.150435 | 1.054888 |
| nvt | species_KE_eV_Z14 | 1.874062 | 1.624948 | 2.075842 | -0.2491145 |
| nvt | macro_16fs_VDOS_overlap_Z8 | 0.8109382 | 0.5991716 | 1 | -0.2117665 |
| nvt | macro_16fs_VDOS_overlap_Z14 | 0.831344 | 0.7313265 | 1 | -0.1000175 |
| nvt | queried_1fs_VDOS_overlap_Z8 | 0.8188459 | 0.6070198 | 1 | -0.2118261 |
| nvt | queried_1fs_VDOS_overlap_Z14 | 0.8339806 | 0.7280543 | 1 | -0.1059262 |
| nvt | RDF_RMSE_14-8 | 0.570315 | 0.1763893 | 0 | -0.3939257 |
| nvt | RDF_14-8_peak_position_A | 1.603341 | 1.603341 | 1.603341 | 0 |
| nvt | RDF_14-8_peak_height | 41.0664 | 36.21299 | 34.76711 | -4.853406 |
| nvt | RDF_RMSE_8-8 | 0.05685032 | 0.2006902 | 0 | 0.1438399 |
| nvt | RDF_8-8_peak_position_A | 2.600975 | 2.624728 | 2.600975 | 0.02375319 |
| nvt | RDF_8-8_peak_height | 8.757685 | 8.185603 | 9.167464 | -0.5720816 |
