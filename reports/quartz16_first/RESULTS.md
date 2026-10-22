# Quartz 16 fs: observed first-run results

Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1`

Model: fixed architecture with learned motion tokens, mode-conditioned recurrent/temporal routing, mid-evolution cross-mode exchange, equivariant segment decoding.
All comparisons below use BKS reference MD. No claim of named physical modes follows from routing weights.

## Training

Completed epoch records: 124. Training wall time: 1.388 h.

## Independent endpoint test

```json
{
  "checkpoint": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1/best_rollout.pt",
  "test_samples": 1000,
  "endpoint": {
    "time_fs": 16,
    "position_MAE_A": 0.001634390757419169,
    "position_RMSE_A": 0.0020959168448181757,
    "velocity_MAE_A_fs": 0.00024996861326508226,
    "velocity_RMSE_A_fs": 0.00032856055277849045
  },
  "normalized_endpoint_loss": 0.005646928784291153,
  "position_velocity_consistency_RMSE_A_fs": 0.00011442304736482255,
  "reference_finite_difference_RMSE_A_fs": 8.725477203009129e-06,
  "join_position_derivative_jump_RMS_A_fs": 0.000570044386408495,
  "join_state_discontinuity": 0.0,
  "two_step_position_RMSE_A": 0.004233009617755802,
  "mode_statistics": {
    "scalar_allocation": [
      0.17231312996149062,
      0.18319448655843734,
      0.19524572598934173,
      0.4492466653585434
    ],
    "vector_allocation": [
      0.14677986586093902,
      0.1467450950741768,
      0.20369901084899902,
      0.5027760355472565
    ],
    "recurrent_route_by_mode": [
      0.5890444965362549,
      0.6047089860439301,
      0.5848731050491333,
      0.07982930752635002
    ],
    "cross_mode_attention": [
      [
        0.03946507593989372,
        0.28353990602493284,
        0.6033198721408845,
        0.07367514938116074
      ],
      [
        0.05309057229757309,
        0.5383382134437561,
        0.36554089391231537,
        0.04303032371401787
      ],
      [
        0.058931020498275755,
        0.19885144460201262,
        0.29901409780979155,
        0.4432034419775009
      ],
      [
        1.845645775756566e-05,
        6.318015561191714e-06,
        0.0004645647864090279,
        0.9995106625556945
      ]
    ],
    "fusion_weights": [
      8.110640454106032e-05,
      5.651364637014922e-05,
      0.015010180179029704,
      0.9848522002696991
    ]
  }
}
```

## Autonomous propagation

| Run | Production ps | Failure | ns/day (including I/O) |
|---|---:|---|---:|
| nve_test_09 | 100 | none recorded | 162.2 |
| nve_test_10 | 100 | none recorded | 165.7 |
| nvt | 15000 | none recorded | 163.3 |

## nve_test_09: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1/nve_test_09.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/raw_nve/test_09/trajectory.h5",
  "stable": true,
  "failure": "",
  "production_available": true,
  "evaluation_phase": "production",
  "duration_ps": 100.0,
  "requested_production_ps": 100.0,
  "minimum_distance_A": 1.4010207653045654,
  "temperature_K": {
    "SMDP": {
      "mean": 700.5481749315527,
      "std": 96.99214487172651,
      "variance": 9407.476166817984,
      "block_mean_std": 33.49538149829002,
      "blocks": 8
    },
    "reference": {
      "mean": 287.90947598928165,
      "std": 12.809218392126452,
      "variance": 164.07607581719057,
      "block_mean_std": 0.2654719089685282,
      "blocks": 8
    },
    "mean_difference": 412.63869894227105,
    "wasserstein": 412.6386989422711
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 14.579009871318236,
      "std": 2.0184899313503877,
      "variance": 4.074301602962892,
      "block_mean_std": 0.6970676892489984,
      "blocks": 8
    },
    "reference": {
      "mean": 5.9916437479890785,
      "std": 0.26657084846581564,
      "variance": 0.07106001725178485,
      "block_mean_std": 0.0055246986858367925,
      "blocks": 8
    },
    "mean_difference": 8.587366123329158,
    "wasserstein": 8.587366123329158
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 10.193822122094808,
        "std": 1.6043782549401335,
        "variance": 2.574029584924748,
        "block_mean_std": 0.46058517445894026,
        "blocks": 8
      },
      "reference": {
        "mean": 3.9952585592974605,
        "std": 0.25656418700238043,
        "variance": 0.06582518205219244,
        "block_mean_std": 0.019453746893190668,
        "blocks": 8
      },
      "mean_difference": 6.198563562797347,
      "wasserstein": 6.198563562797347
    },
    "14": {
      "SMDP": {
        "mean": 4.385187749223429,
        "std": 0.7768368826186639,
        "variance": 0.6034755421966839,
        "block_mean_std": 0.30587388100168317,
        "blocks": 8
      },
      "reference": {
        "mean": 1.9963851886916182,
        "std": 0.19749555049054793,
        "variance": 0.03900449246356456,
        "block_mean_std": 0.02346104770532466,
        "blocks": 8
      },
      "mean_difference": 2.3888025605318113,
      "wasserstein": 2.38880256053181
    }
  },
  "energy_evaluation": "Identical offline BKS on exact sampled configurations for candidate and reference; raw coul/dsf convention",
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3127.568626979543,
      "std": 1.9678731021825988,
      "variance": 3.8725245462937647,
      "block_mean_std": 0.706476745820367,
      "blocks": 8
    },
    "reference": {
      "mean": -3136.011974581337,
      "std": 0.26786188351384876,
      "variance": 0.07174998863958669,
      "block_mean_std": 0.007124675429953361,
      "blocks": 8
    },
    "mean_difference": 8.443347601794358,
    "wasserstein": 8.443347601794141
  },
  "total_energy_drift_meV_atom_ps": {
    "SMDP": 0.1389062866730689,
    "reference": 0.0005761529653860687
  },
  "energy_change_eV": {
    "SMDP": 17.429389199546677,
    "reference": 0.004781890222602669
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.000909770088241403,
    "std": 6.826119223539151e-05,
    "variance": 4.659590365397074e-09,
    "block_mean_std": 2.3067017575215566e-05,
    "blocks": 8
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.7344079556206102,
          "block_overlap": {
            "mean": 0.7306447525711346,
            "std": 0.014104948453958798,
            "variance": 0.00019894957088883468,
            "block_mean_std": 0.014104948453958798,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9766100112697568
        },
        "14": {
          "overlap": 0.7947465418966826,
          "block_overlap": {
            "mean": 0.7908899893534488,
            "std": 0.011312143699623717,
            "variance": 0.00012796459508093656,
            "block_mean_std": 0.011312143699623717,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9794413408438695
        }
      }
    },
    "queried_1fs": {
      "nyquist_THz": 500.0,
      "elements": {
        "8": {
          "overlap": 0.7331087899491009,
          "block_overlap": {
            "mean": 0.7294408785027446,
            "std": 0.013353546150799008,
            "variance": 0.00017831719480151898,
            "block_mean_std": 0.013353546150799008,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9760548538076976
        },
        "14": {
          "overlap": 0.7859661533258806,
          "block_overlap": {
            "mean": 0.7819346909068366,
            "std": 0.011552078151592165,
            "variance": 0.00013345050962049304,
            "block_mean_std": 0.011552078151592165,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9782485236482704
        }
      }
    }
  },
  "RDF": {
    "14-8": {
      "RMSE": 1.3086306578139808,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 22.73502860566931,
        "reference": 35.364217861851046,
        "difference": -12.629189256181736
      },
      "block_curve_RMSE": {
        "mean": 1.31112067071978,
        "std": 0.11268548737371287,
        "variance": 0.012698019064651202,
        "block_mean_std": 0.11268548737371287,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.39486326453309467,
      "peak_position_A": {
        "SMDP": 2.624727889102222,
        "reference": 2.6009746955356867,
        "difference": 0.02375319356653538
      },
      "peak_height": {
        "SMDP": 6.692429582050862,
        "reference": 9.24784183169206,
        "difference": -2.555412249641198
      },
      "block_curve_RMSE": {
        "mean": 0.39741026715843863,
        "std": 0.02259575615544385,
        "variance": 0.0005105681962362786,
        "block_mean_std": 0.02259575615544385,
        "blocks": 8
      }
    }
  }
}
```

## nve_test_10: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1/nve_test_10.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/raw_nve/test_10/trajectory.h5",
  "stable": true,
  "failure": "",
  "production_available": true,
  "evaluation_phase": "production",
  "duration_ps": 100.0,
  "requested_production_ps": 100.0,
  "minimum_distance_A": 1.4166399240493774,
  "temperature_K": {
    "SMDP": {
      "mean": 707.7278678269296,
      "std": 98.20230314542871,
      "variance": 9643.69234306668,
      "block_mean_std": 29.14061323732776,
      "blocks": 8
    },
    "reference": {
      "mean": 302.7117153987662,
      "std": 13.544567609039127,
      "variance": 183.4553117158319,
      "block_mean_std": 0.47524526242154613,
      "blocks": 8
    },
    "mean_difference": 405.01615242816337,
    "wasserstein": 405.01615242816337
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 14.728425453772589,
      "std": 2.0436743655541965,
      "variance": 4.176604912423347,
      "block_mean_std": 0.6064412173863386,
      "blocks": 8
    },
    "reference": {
      "mean": 6.299691077481552,
      "std": 0.28187409794367413,
      "variance": 0.07945300709155999,
      "block_mean_std": 0.009890262540214223,
      "blocks": 8
    },
    "mean_difference": 8.428734376291036,
    "wasserstein": 8.428734376291036
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 10.361380238348001,
        "std": 1.5914348324743397,
        "variance": 2.5326648260126294,
        "block_mean_std": 0.4321165988779356,
        "blocks": 8
      },
      "reference": {
        "mean": 4.2171512847601225,
        "std": 0.26756697380678646,
        "variance": 0.07159208547212154,
        "block_mean_std": 0.02948678633418148,
        "blocks": 8
      },
      "mean_difference": 6.144228953587879,
      "wasserstein": 6.144228953587881
    },
    "14": {
      "SMDP": {
        "mean": 4.367045215424587,
        "std": 0.7697611556440551,
        "variance": 0.5925322367384713,
        "block_mean_std": 0.2008811855076454,
        "blocks": 8
      },
      "reference": {
        "mean": 2.082539792721431,
        "std": 0.21225893510662974,
        "variance": 0.045053855532600454,
        "block_mean_std": 0.029218059545168873,
        "blocks": 8
      },
      "mean_difference": 2.284505422703156,
      "wasserstein": 2.2845054227031563
    }
  },
  "energy_evaluation": "Identical offline BKS on exact sampled configurations for candidate and reference; raw coul/dsf convention",
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3127.374445120835,
      "std": 2.021346684704532,
      "variance": 4.085842419766004,
      "block_mean_std": 0.5806314277834748,
      "blocks": 8
    },
    "reference": {
      "mean": -3135.704982468285,
      "std": 0.2832842864854356,
      "variance": 0.08024998696956234,
      "block_mean_std": 0.011870085262314306,
      "blocks": 8
    },
    "mean_difference": 8.33053734744999,
    "wasserstein": 8.330537347450408
  },
  "total_energy_drift_meV_atom_ps": {
    "SMDP": 0.1447015814070018,
    "reference": -0.00047076049092033913
  },
  "energy_change_eV": {
    "SMDP": 19.111451112087707,
    "reference": 0.003283001652562234
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.0009105021552004098,
    "std": 6.488958771888343e-05,
    "variance": 4.210658594326667e-09,
    "block_mean_std": 1.6825926647454108e-05,
    "blocks": 8
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.7301604023565964,
          "block_overlap": {
            "mean": 0.7268645945012441,
            "std": 0.006208110060327355,
            "variance": 3.8540630521137716e-05,
            "block_mean_std": 0.006208110060327355,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9739188229434805
        },
        "14": {
          "overlap": 0.7958643828733072,
          "block_overlap": {
            "mean": 0.7935557461807807,
            "std": 0.0036204125583150673,
            "variance": 1.3107387092405452e-05,
            "block_mean_std": 0.0036204125583150673,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9758987867209087
        }
      }
    },
    "queried_1fs": {
      "nyquist_THz": 500.0,
      "elements": {
        "8": {
          "overlap": 0.7348608381104262,
          "block_overlap": {
            "mean": 0.7311623760629767,
            "std": 0.0077110368249616555,
            "variance": 5.9460088915914734e-05,
            "block_mean_std": 0.0077110368249616555,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9729947939441181
        },
        "14": {
          "overlap": 0.7921053266822665,
          "block_overlap": {
            "mean": 0.7884817507052769,
            "std": 0.003972677684517986,
            "variance": 1.5782167985067184e-05,
            "block_mean_std": 0.003972677684517986,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9755400287991051
        }
      }
    }
  },
  "RDF": {
    "14-8": {
      "RMSE": 1.2134816020435613,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 23.024969834993676,
        "reference": 34.6474984533892,
        "difference": -11.62252861839552
      },
      "block_curve_RMSE": {
        "mean": 1.216054923774208,
        "std": 0.0885814407432735,
        "variance": 0.007846671644154073,
        "block_mean_std": 0.0885814407432735,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.38724508593335166,
      "peak_position_A": {
        "SMDP": 2.624727889102222,
        "reference": 2.6009746955356867,
        "difference": 0.02375319356653538
      },
      "peak_height": {
        "SMDP": 6.654586666109493,
        "reference": 9.071304469266204,
        "difference": -2.416717803156711
      },
      "block_curve_RMSE": {
        "mean": 0.3895832059977467,
        "std": 0.028877341147791433,
        "variance": 0.0008339008317659283,
        "block_mean_std": 0.028877341147791433,
        "blocks": 8
      }
    }
  }
}
```

## nvt: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1/nvt.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/physical_reference/nvt_15ns/production.h5",
  "stable": true,
  "failure": "",
  "production_available": true,
  "evaluation_phase": "production",
  "duration_ps": 15000.0,
  "requested_production_ps": 15000.0,
  "minimum_distance_A": 1.4701242446899414,
  "temperature_K": {
    "SMDP": {
      "mean": 361.3443900516864,
      "std": 28.766068989284378,
      "variance": 827.4867250962684,
      "block_mean_std": 1.5082292329673845,
      "blocks": 8
    },
    "reference": {
      "mean": 299.1839998544485,
      "std": 18.658345123931916,
      "variance": 348.13384276375393,
      "block_mean_std": 1.0521074979856517,
      "blocks": 8
    },
    "mean_difference": 62.160390197237916,
    "wasserstein": 62.160390197237966
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 7.5198874510289295,
      "std": 0.5986466295408935,
      "variance": 0.3583777870606718,
      "block_mean_std": 0.03138754715589769,
      "blocks": 8
    },
    "reference": {
      "mean": 6.226276283775414,
      "std": 0.38829620499809836,
      "variance": 0.15077394281592524,
      "block_mean_std": 0.021895261664652242,
      "blocks": 8
    },
    "mean_difference": 1.2936111672535153,
    "wasserstein": 1.2936111672535158
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 6.0536095824359055,
        "std": 0.5431864902980234,
        "variance": 0.29505156324228465,
        "block_mean_std": 0.032633965783491765,
        "blocks": 8
      },
      "reference": {
        "mean": 4.1504345011315,
        "std": 0.3216096183131783,
        "variance": 0.10343274659154823,
        "block_mean_std": 0.020163058479425323,
        "blocks": 8
      },
      "mean_difference": 1.903175081304406,
      "wasserstein": 1.9031750813044044
    },
    "14": {
      "SMDP": {
        "mean": 1.4662778685930251,
        "std": 0.29477940014082815,
        "variance": 0.08689489474738649,
        "block_mean_std": 0.01956384308064176,
        "blocks": 8
      },
      "reference": {
        "mean": 2.075841782643914,
        "std": 0.22705827190739697,
        "variance": 0.05155545884157342,
        "block_mean_std": 0.010641413793087697,
        "blocks": 8
      },
      "mean_difference": -0.609563914050889,
      "wasserstein": 0.6118861661716248
    }
  },
  "energy_evaluation": "Identical offline BKS on exact sampled configurations for candidate and reference; raw coul/dsf convention",
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3134.6822155737327,
      "std": 0.5775837326127798,
      "variance": 0.3336029681789111,
      "block_mean_std": 0.026143096232558782,
      "blocks": 8
    },
    "reference": {
      "mean": -3135.7729840013926,
      "std": 0.40106456900913406,
      "variance": 0.16085278851448248,
      "block_mean_std": 0.027322190507256355,
      "blocks": 8
    },
    "mean_difference": 1.0907684276598957,
    "wasserstein": 1.0907684276596314
  },
  "total_energy_drift_meV_atom_ps": {
    "SMDP": -2.1249821436384444e-05,
    "reference": -8.948068398362613e-06
  },
  "energy_change_eV": {
    "SMDP": -0.8342772727974079,
    "reference": -1.2183689951611996
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.000621099309282695,
    "std": 4.2135798361096114e-05,
    "variance": 1.7754255035269502e-09,
    "block_mean_std": 9.049547892783418e-07,
    "blocks": 8
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.44967180080085783,
          "block_overlap": {
            "mean": 0.4468702158952285,
            "std": 0.02137603211290212,
            "variance": 0.0004569347488918228,
            "block_mean_std": 0.0020047003671945286,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9987517641216376
        },
        "14": {
          "overlap": 0.6281193433339972,
          "block_overlap": {
            "mean": 0.6222274093004803,
            "std": 0.014029754702923313,
            "variance": 0.00019683401702419882,
            "block_mean_std": 0.001566866849315211,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9985316190610183
        }
      }
    },
    "queried_1fs": {
      "nyquist_THz": 500.0,
      "elements": {
        "8": {
          "overlap": 0.45773409264546233,
          "block_overlap": {
            "mean": 0.4545754581506839,
            "std": 0.021782956549417514,
            "variance": 0.0004744971960338113,
            "block_mean_std": 0.0020832797848007747,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9986130057655248
        },
        "14": {
          "overlap": 0.6196627440373232,
          "block_overlap": {
            "mean": 0.6135743693174855,
            "std": 0.014867358602706214,
            "variance": 0.00022103835182146246,
            "block_mean_std": 0.0016963935315100192,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9984999858738758
        }
      }
    }
  },
  "RDF": {
    "14-8": {
      "RMSE": 0.24015275193869992,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 36.92970911985332,
        "reference": 34.76711117175737,
        "difference": 2.16259794809595
      },
      "block_curve_RMSE": {
        "mean": 0.24895414075626587,
        "std": 0.027294282981193903,
        "variance": 0.0007449778834574912,
        "block_mean_std": 0.027294282981193903,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.3096487856862614,
      "peak_position_A": {
        "SMDP": 2.6484810826687584,
        "reference": 2.6009746955356867,
        "difference": 0.04750638713307165
      },
      "peak_height": {
        "SMDP": 7.931827561015302,
        "reference": 9.167464113664485,
        "difference": -1.2356365526491828
      },
      "block_curve_RMSE": {
        "mean": 0.3132382769302204,
        "std": 0.011006873363095134,
        "variance": 0.00012115126123121318,
        "block_mean_std": 0.011006873363095134,
        "blocks": 8
      }
    }
  }
}
```

## Observed mechanism and cost limitations

Final validation mean fusion weights: [7.969021870673755e-05, 5.523974011926945e-05, 0.01561988703086316, 0.984245220979182].
Fusion is dominated by one mode (>98%). This run does not establish useful heterogeneous modal specialization; no balancing regularizer was added to disguise it.
Soft routing evaluates both recurrent and temporal operators. Inference cost includes all 16 time queries; a learned route does not save expert computation. No matched-hardware BKS speedup is claimed without a measured baseline.

## Interpretation boundaries

Only completed production frames are production evidence. Failed thermalization statistics are diagnostics, not 15 ns NVT results.
16 fs spectra compare identically downsampled outputs. Queried 1 fs spectra compare dense velocities separately and include endpoint thermostat kicks; output cadence alone does not establish high-frequency fidelity.
This experiment stops after these evaluations; there is no automatic architecture change or second training run.
