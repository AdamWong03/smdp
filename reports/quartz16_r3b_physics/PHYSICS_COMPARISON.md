# R3-B 固定 checkpoint 正式物理评价

全部请求完成：True
未重新训练或筛选模型。状态按实际轨迹和后处理记录，而不是目标长度判断。
轨迹/后处理状态：
```json
{
  "nve_test_09": {
    "trajectory_completed": true,
    "physical_analysis_completed": true,
    "production_ps": 100.0,
    "failure_reason": null,
    "analysis_failure_reason": {}
  },
  "nve_test_10": {
    "trajectory_completed": true,
    "physical_analysis_completed": true,
    "production_ps": 100.0,
    "failure_reason": null,
    "analysis_failure_reason": {}
  },
  "nvt": {
    "trajectory_completed": true,
    "physical_analysis_completed": true,
    "production_ps": 15000.0,
    "failure_reason": null,
    "analysis_failure_reason": {}
  }
}
```

## R1 / R2 / R3-B / BKS 同协议对照

| Run | Metric | R1 | R2 | R3-B | BKS |
|---|---|---:|---:|---:|---:|
| nve_test_09 | temperature_K_mean | 700.54817 | 254.28398 | 501.02695 | 287.90948 |
| nve_test_09 | temperature_K_std | 96.992145 | 44.551643 | 71.855615 | 12.809218 |
| nve_test_09 | kinetic_energy_eV_mean | 14.57901 | 5.2918683 | 10.426802 | 5.9916437 |
| nve_test_09 | kinetic_energy_eV_std | 2.0184899 | 0.927158 | 1.4953771 | 0.26657085 |
| nve_test_09 | potential_energy_eV_mean | -3127.5686 | -3136.7908 | -3131.5528 | -3136.012 |
| nve_test_09 | potential_energy_eV_std | 1.9678731 | 0.88018563 | 1.487002 | 0.26786188 |
| nve_test_09 | potential_Wasserstein_eV | 8.4433476 | 0.84857051 | 4.4600679 | 0 |
| nve_test_09 | energy_change_eV | 17.429389 | 0.13941236 | 10.712032 | 0.0047818902 |
| nve_test_09 | total_energy_drift_meV_atom_ps | 0.13890629 | 0.11258551 | 0.077560592 | 0.00057615297 |
| nve_test_09 | minimum_distance_A | 1.4010208 | 1.4896187 | 1.4312519 | 未测/不可用 |
| nve_test_09 | species_KE_Z8 | 10.193822 | 3.5633629 | 7.6367635 | 3.9952586 |
| nve_test_09 | species_KE_Z14 | 4.3851877 | 1.7285054 | 2.7900382 | 1.9963852 |
| nve_test_09 | macro_16fs_VDOS_Z8 | 0.73440796 | 0.68830819 | 0.71003917 | 1 |
| nve_test_09 | macro_16fs_VDOS_Z14 | 0.79474654 | 0.71492747 | 0.79646657 | 1 |
| nve_test_09 | queried_1fs_VDOS_Z8 | 0.73310879 | 0.70150935 | 0.70786212 | 1 |
| nve_test_09 | queried_1fs_VDOS_Z14 | 0.78596615 | 0.71467627 | 0.78622344 | 1 |
| nve_test_09 | RDF_RMSE_14-8 | 1.3086307 | 0.68632989 | 0.90849702 | 0 |
| nve_test_09 | RDF_14-8_peak_position_A | 1.6033406 | 1.6033406 | 1.6033406 | 1.6033406 |
| nve_test_09 | RDF_14-8_peak_height | 22.735029 | 43.108424 | 26.373169 | 35.364218 |
| nve_test_09 | RDF_RMSE_8-8 | 0.39486326 | 0.07114615 | 0.30385387 | 0 |
| nve_test_09 | RDF_8-8_peak_position_A | 2.6247279 | 2.6247279 | 2.6247279 | 2.6009747 |
| nve_test_09 | RDF_8-8_peak_height | 6.6924296 | 8.9954413 | 7.3984703 | 9.2478418 |
| nve_test_09 | ns_day_end_to_end | 162.226 | 109.008 | 93.385161 | 未测/不可用 |
| nve_test_09 | model_seconds | 31.548183 | 42.277807 | 73.473364 | 未测/不可用 |
| nve_test_09 | neighbor_seconds | 10.399001 | 11.379825 | 9.1255463 | 未测/不可用 |
| nve_test_09 | elapsed_s | 53.259035 | 79.260235 | 92.520053 | 未测/不可用 |
| nve_test_09 | peak_gpu_bytes | 1.2425165e+08 | 1.2445952e+08 | 1.1007227e+09 | 未测/不可用 |
| nve_test_09 | total_energy_RMS_change_eV | 17.345453 | 2.0032197 | 9.2543785 | 0.0041110826 |
| nve_test_09 | total_energy_min_change_eV | -0.87452038 | -5.1818466 | -1.4172549 | -0.011302005 |
| nve_test_09 | total_energy_max_change_eV | 23.757604 | 1.5083054 | 14.685974 | 0.012600449 |
| nve_test_10 | temperature_K_mean | 707.72787 | 274.82347 | 506.18917 | 302.71172 |
| nve_test_10 | temperature_K_std | 98.202303 | 63.229609 | 75.042427 | 13.544568 |
| nve_test_10 | kinetic_energy_eV_mean | 14.728425 | 5.7193127 | 10.534232 | 6.2996911 |
| nve_test_10 | kinetic_energy_eV_std | 2.0436744 | 1.3158625 | 1.5616974 | 0.2818741 |
| nve_test_10 | potential_energy_eV_mean | -3127.3744 | -3136.3762 | -3131.4161 | -3135.705 |
| nve_test_10 | potential_energy_eV_std | 2.0213467 | 1.284291 | 1.547572 | 0.28328429 |
| nve_test_10 | potential_Wasserstein_eV | 8.3305373 | 0.99838929 | 4.2897846 | 0 |
| nve_test_10 | energy_change_eV | 19.111451 | -2.2499796 | 8.5709227 | 0.0032830016 |
| nve_test_10 | total_energy_drift_meV_atom_ps | 0.14470158 | 0.21577484 | 0.10208091 | -0.00047076049 |
| nve_test_10 | minimum_distance_A | 1.4166399 | 1.5008293 | 1.4508488 | 未测/不可用 |
| nve_test_10 | species_KE_Z8 | 10.36138 | 3.9602741 | 7.7439933 | 4.2171513 |
| nve_test_10 | species_KE_Z14 | 4.3670452 | 1.7590386 | 2.7902385 | 2.0825398 |
| nve_test_10 | macro_16fs_VDOS_Z8 | 0.7301604 | 0.6608274 | 0.7094248 | 1 |
| nve_test_10 | macro_16fs_VDOS_Z14 | 0.79586438 | 0.71908932 | 0.79724988 | 1 |
| nve_test_10 | queried_1fs_VDOS_Z8 | 0.73486084 | 0.67529043 | 0.71191259 | 1 |
| nve_test_10 | queried_1fs_VDOS_Z14 | 0.79210533 | 0.71628244 | 0.78936663 | 1 |
| nve_test_10 | RDF_RMSE_14-8 | 1.2134816 | 0.70074396 | 0.84195429 | 0 |
| nve_test_10 | RDF_14-8_peak_position_A | 1.6033406 | 1.6033406 | 1.6033406 | 1.6033406 |
| nve_test_10 | RDF_14-8_peak_height | 23.02497 | 42.256781 | 26.189444 | 34.647498 |
| nve_test_10 | RDF_RMSE_8-8 | 0.38724509 | 0.07538397 | 0.28337405 | 0 |
| nve_test_10 | RDF_8-8_peak_position_A | 2.6247279 | 2.6247279 | 2.6247279 | 2.6009747 |
| nve_test_10 | RDF_8-8_peak_height | 6.6545867 | 8.9129077 | 7.3310738 | 9.0713045 |
| nve_test_10 | ns_day_end_to_end | 165.67991 | 116.30902 | 92.196716 | 未测/不可用 |
| nve_test_10 | model_seconds | 30.912048 | 39.748232 | 74.253339 | 未测/不可用 |
| nve_test_10 | neighbor_seconds | 10.347189 | 11.754453 | 9.1955723 | 未测/不可用 |
| nve_test_10 | elapsed_s | 52.14875 | 74.284869 | 93.712665 | 未测/不可用 |
| nve_test_10 | peak_gpu_bytes | 1.2425165e+08 | 1.2445952e+08 | 1.099412e+09 | 未测/不可用 |
| nve_test_10 | total_energy_RMS_change_eV | 17.118022 | 2.3481589 | 8.9041998 | 0.0058801147 |
| nve_test_10 | total_energy_min_change_eV | -0.84324649 | -6.1725363 | -1.609974 | -0.017883033 |
| nve_test_10 | total_energy_max_change_eV | 25.410032 | 2.3875689 | 14.107521 | 0.0067940871 |
| nvt | temperature_K_mean | 361.34439 | 280.74781 | 319.46668 | 299.184 |
| nvt | temperature_K_std | 28.766069 | 19.711808 | 26.000936 | 18.658345 |
| nvt | kinetic_energy_eV_mean | 7.5198875 | 5.8426033 | 6.6483764 | 6.2262763 |
| nvt | kinetic_energy_eV_std | 0.59864663 | 0.41021968 | 0.54110183 | 0.3882962 |
| nvt | potential_energy_eV_mean | -3134.6822 | -3136.1637 | -3135.3281 | -3135.773 |
| nvt | potential_energy_eV_std | 0.57758373 | 0.40318902 | 0.5288545 | 0.40106457 |
| nvt | potential_Wasserstein_eV | 1.0907684 | 0.3906676 | 0.44485612 | 0 |
| nvt | energy_change_eV | -0.83427727 | 0.11373545 | -0.25760064 | -1.218369 |
| nvt | total_energy_drift_meV_atom_ps | -2.1249821e-05 | -3.2992658e-06 | -6.1916734e-06 | -8.9480684e-06 |
| nvt | minimum_distance_A | 1.4701242 | 1.478557 | 1.4571613 | 未测/不可用 |
| nvt | species_KE_Z8 | 6.0536096 | 3.9685411 | 5.0234287 | 4.1504345 |
| nvt | species_KE_Z14 | 1.4662779 | 1.8740622 | 1.6249477 | 2.0758418 |
| nvt | macro_16fs_VDOS_Z8 | 0.4496718 | 0.81093815 | 0.59917161 | 1 |
| nvt | macro_16fs_VDOS_Z14 | 0.62811934 | 0.83134402 | 0.73132649 | 1 |
| nvt | queried_1fs_VDOS_Z8 | 0.45773409 | 0.81884594 | 0.60701984 | 1 |
| nvt | queried_1fs_VDOS_Z14 | 0.61966274 | 0.83398057 | 0.72805432 | 1 |
| nvt | RDF_RMSE_14-8 | 0.24015275 | 0.57031499 | 0.17638926 | 0 |
| nvt | RDF_14-8_peak_position_A | 1.6033406 | 1.6033406 | 1.6033406 | 1.6033406 |
| nvt | RDF_14-8_peak_height | 36.929709 | 41.066395 | 36.21299 | 34.767111 |
| nvt | RDF_RMSE_8-8 | 0.30964879 | 0.056850322 | 0.20069019 | 0 |
| nvt | RDF_8-8_peak_position_A | 2.6484811 | 2.6009747 | 2.6247279 | 2.6009747 |
| nvt | RDF_8-8_peak_height | 7.9318276 | 8.7576846 | 8.1856029 | 9.1674641 |
| nvt | ns_day_end_to_end | 163.25223 | 104.90991 | 29.863371 | 未测/不可用 |
| nvt | model_seconds | 5252.0983 | 7269.2626 | 28282.049 | 未测/不可用 |
| nvt | neighbor_seconds | 1640.269 | 1906.3447 | 3359.5299 | 未测/不可用 |
| nvt | elapsed_s | 8732.4994 | 13588.802 | 47737.411 | 未测/不可用 |
| nvt | peak_gpu_bytes | 1.2425011e+08 | 1.2446106e+08 | 1.101076e+09 | 未测/不可用 |

## 口径与结论边界

能量和温度对照沿用 2000 个均匀生产帧，RDF 使用 512 帧；完整时间范围的 KE/PE/总能量曲线同时保留，不能将首末能量抵消视为守恒。时间窗指标见 nve_window_comparison.json。
VDOS 使用既有窗口/归一化，16 fs 自主端点与 1 fs 神经查询分别报告；1 fs 采样本身不证明高频正确。
NVT 维持 300 K、100 fs CSVR 耦合时间、1.5 ns 热化、15 ns 生产；边界恒温间隔16 fs，不是参考MD的1 fs。thermostat.json记录网络与恒温器的动能交换。
所有数值稳定/完成状态仅表示通过原有停止条件，不等于能量守恒或正确平衡分布。若未完成，短轨迹统计不可作为同长度正式对照。
生产吞吐包含图更新、网络、恒温与输出；另存 production_efficiency_benchmark.json 的端点/密集查询短时计时，不混作端到端15 ns吞吐。历史运行共享GPU负载可能不同，不据此宣称严格加速。
两条NVE是独立初态；NVT时间块波动并非独立模拟置信区间。逐项物理改进与退步均在表中保留。本轮不自动进入R4或调节恒温器。
