# Quartz 16 fs: observed results

Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition`

Model: fixed architecture with learned motion tokens, mode-conditioned recurrent/temporal routing, mid-evolution cross-mode exchange, equivariant segment decoding.
All comparisons below use BKS reference MD. No claim of named physical modes follows from routing weights.

## Training

Completed epoch records: 124. Training wall time: 5.025 h.

## Independent endpoint test

```json
{
  "checkpoint": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition/best_rollout.pt",
  "test_samples": 1000,
  "endpoint": {
    "time_fs": 16,
    "position_MAE_A": 0.0019615453304722905,
    "position_RMSE_A": 0.0024997697297838542,
    "velocity_MAE_A_fs": 0.00026319567242171616,
    "velocity_RMSE_A_fs": 0.00034254794762157774
  },
  "checkpoint_sha256": "667dcbc8a3989b4eea1204e5455724e0b1a9105a5574cb6df0ffddf5579b58ba",
  "normalized_endpoint_loss": 0.006751076446077853,
  "position_velocity_consistency_RMSE_A_fs": 9.084611258007447e-05,
  "reference_finite_difference_RMSE_A_fs": 8.725477203009129e-06,
  "join_position_derivative_jump_RMS_A_fs": 0.0004751207831153209,
  "join_state_discontinuity": 0.0,
  "two_step_position_RMSE_A": 0.005025741786463624,
  "reference_join_derivative_jump_RMS_A_fs": 0.0004941729594609555,
  "multi_step": {
    "2": {
      "time_fs": 32,
      "samples": 991,
      "position_RMSE_A": 0.005025592846123881,
      "velocity_RMSE_A_fs": 0.00043272341439654224
    },
    "3": {
      "time_fs": 48,
      "samples": 991,
      "position_RMSE_A": 0.007370765178337023,
      "velocity_RMSE_A_fs": 0.0005044800413862824
    },
    "4": {
      "time_fs": 64,
      "samples": 991,
      "position_RMSE_A": 0.009606192894497245,
      "velocity_RMSE_A_fs": 0.0005807930572105109
    },
    "5": {
      "time_fs": 80,
      "samples": 991,
      "position_RMSE_A": 0.011857378337272875,
      "velocity_RMSE_A_fs": 0.0006449987119927122
    },
    "6": {
      "time_fs": 96,
      "samples": 991,
      "position_RMSE_A": 0.01407063709894546,
      "velocity_RMSE_A_fs": 0.0007041779063304617
    },
    "7": {
      "time_fs": 112,
      "samples": 991,
      "position_RMSE_A": 0.016225655088870732,
      "velocity_RMSE_A_fs": 0.0007551066792456577
    },
    "8": {
      "time_fs": 128,
      "samples": 991,
      "position_RMSE_A": 0.018310923820276385,
      "velocity_RMSE_A_fs": 0.0008141269257066777
    },
    "9": {
      "time_fs": 144,
      "samples": 991,
      "position_RMSE_A": 0.020437957093699496,
      "velocity_RMSE_A_fs": 0.0008733836837664364
    },
    "10": {
      "time_fs": 160,
      "samples": 991,
      "position_RMSE_A": 0.022606315483365294,
      "velocity_RMSE_A_fs": 0.0009321828504448028
    },
    "11": {
      "time_fs": 176,
      "samples": 991,
      "position_RMSE_A": 0.024795609874762176,
      "velocity_RMSE_A_fs": 0.000989693769337298
    },
    "12": {
      "time_fs": 192,
      "samples": 991,
      "position_RMSE_A": 0.026992916463347862,
      "velocity_RMSE_A_fs": 0.0010479955695972055
    },
    "13": {
      "time_fs": 208,
      "samples": 991,
      "position_RMSE_A": 0.02922518456158935,
      "velocity_RMSE_A_fs": 0.0011018310094221616
    },
    "14": {
      "time_fs": 224,
      "samples": 991,
      "position_RMSE_A": 0.03145530448478804,
      "velocity_RMSE_A_fs": 0.00115632441810164
    },
    "15": {
      "time_fs": 240,
      "samples": 991,
      "position_RMSE_A": 0.033697368202296435,
      "velocity_RMSE_A_fs": 0.0012041983845672531
    },
    "16": {
      "time_fs": 256,
      "samples": 991,
      "position_RMSE_A": 0.035909956309655565,
      "velocity_RMSE_A_fs": 0.0012555184626971976
    }
  },
  "mode_statistics": {
    "scalar_allocation": [
      0.10767583498358727,
      0.2414672136902809,
      0.32081881058216094,
      0.3300381489992142
    ],
    "vector_allocation": [
      0.05296261695027352,
      0.35397200763225556,
      0.1886636371612549,
      0.40440174722671507
    ],
    "recurrent_route_by_mode": [
      0.5589888696670532,
      0.9687325699329377,
      0.00445296262204647,
      0.8096654090881348
    ],
    "cross_mode_attention": [
      [
        0.0,
        0.00041950208554044363,
        0.6379025120735169,
        0.3616779885292053
      ],
      [
        5.670709778087258e-07,
        0.0,
        7.515741534007247e-06,
        0.99999191904068
      ],
      [
        2.199104854662437e-05,
        3.668083883894724e-06,
        0.0,
        0.9999743402004242
      ],
      [
        2.6991890872523072e-06,
        0.33333055436611175,
        0.6666667559146882,
        0.0
      ]
    ],
    "mode_delta_q_rms": [
      0.0022209939314052464,
      0.017663043789565562,
      0.015842137143015862,
      0.020940372355282306
    ],
    "mode_delta_q_rms_share": [
      0.03918396931886673,
      0.3117607311010361,
      0.2795344398021698,
      0.369520858168602
    ],
    "mode_delta_q_signed_projection": [
      0.04171104855835438,
      0.26918408155441287,
      0.2821934667825699,
      0.40691140401363374
    ],
    "mode_delta_v_rms": [
      0.00024558871873887254,
      0.00205803736532107,
      0.0018701027911156415,
      0.0035319875720888376
    ],
    "mode_delta_v_rms_share": [
      0.03187212947756052,
      0.26711683678627013,
      0.24269377756118773,
      0.4583172538280487
    ],
    "mode_delta_v_signed_projection": [
      0.03294657191634178,
      0.2313993564248085,
      0.1360689259171486,
      0.5995851404666901
    ]
  }
}
```

## Autonomous propagation

| Run | Production ps | Failure | ns/day (including I/O) |
|---|---:|---|---:|
| nve_test_09 | 100 | none recorded | 93.39 |
| nve_test_10 | 100 | none recorded | 92.2 |
| nvt | 15000 | none recorded | 29.86 |

## nve_test_09: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition/nve_test_09.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/raw_nve/test_09/trajectory.h5",
  "production_available": true,
  "trajectory_completed": true,
  "stable": true,
  "failure": "",
  "evaluation_phase": "production",
  "duration_ps": 100.0,
  "requested_production_ps": 100.0,
  "temperature_K": {
    "SMDP": {
      "mean": 501.0269520801469,
      "std": 71.85561484788425,
      "variance": 5163.229385167485,
      "block_mean_std": 22.91204525230041,
      "blocks": 8
    },
    "reference": {
      "mean": 287.90947598928165,
      "std": 12.809218392126452,
      "variance": 164.07607581719057,
      "block_mean_std": 0.2654719089685282,
      "blocks": 8
    },
    "mean_difference": 213.11747609086524,
    "wasserstein": 213.16136023502494
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 10.426801669830397,
      "std": 1.4953771284598665,
      "variance": 2.2361527563208763,
      "block_mean_std": 0.47681936212026327,
      "blocks": 8
    },
    "reference": {
      "mean": 5.991643747989079,
      "std": 0.26657084846581564,
      "variance": 0.07106001725178485,
      "block_mean_std": 0.0055246986858367925,
      "blocks": 8
    },
    "mean_difference": 4.435157921841317,
    "wasserstein": 4.436071188614099
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 7.636763456395682,
        "std": 1.2592824506406557,
        "variance": 1.5857922904915354,
        "block_mean_std": 0.3441999090429332,
        "blocks": 8
      },
      "reference": {
        "mean": 3.99525855929746,
        "std": 0.25656418700238043,
        "variance": 0.06582518205219244,
        "block_mean_std": 0.019453746893190658,
        "blocks": 8
      },
      "mean_difference": 3.6415048970982222,
      "wasserstein": 3.641504897098222
    },
    "14": {
      "SMDP": {
        "mean": 2.7900382134347117,
        "std": 0.41944113032271024,
        "variance": 0.1759308618063928,
        "block_mean_std": 0.1456161767634691,
        "blocks": 8
      },
      "reference": {
        "mean": 1.9963851886916182,
        "std": 0.19749555049054793,
        "variance": 0.03900449246356456,
        "block_mean_std": 0.023461047705324648,
        "blocks": 8
      },
      "mean_difference": 0.7936530247430935,
      "wasserstein": 0.794215648160279
    }
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.0006711774299019485,
    "std": 4.970060286573873e-05,
    "variance": 2.4701499252178767e-09,
    "block_mean_std": 1.614497819519286e-05,
    "blocks": 8
  },
  "minimum_distance_A": 1.431251883506775,
  "thermodynamic_sampling": "2000 uniform production frames, matching R1",
  "RDF": {
    "14-8": {
      "RMSE": 0.9084970235229209,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 26.373169047554665,
        "reference": 35.364217861851046,
        "difference": -8.99104881429638
      },
      "block_curve_RMSE": {
        "mean": 0.911743793824295,
        "std": 0.08945483072070665,
        "variance": 0.008002166739270282,
        "block_mean_std": 0.08945483072070665,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.30385386831285877,
      "peak_position_A": {
        "SMDP": 2.624727889102222,
        "reference": 2.6009746955356867,
        "difference": 0.02375319356653538
      },
      "peak_height": {
        "SMDP": 7.3984702708999,
        "reference": 9.24784183169206,
        "difference": -1.84937156079216
      },
      "block_curve_RMSE": {
        "mean": 0.3072995134449491,
        "std": 0.0257666779177998,
        "variance": 0.0006639216909196319,
        "block_mean_std": 0.0257666779177998,
        "blocks": 8
      }
    }
  },
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3131.5528131456554,
      "std": 1.487002012596303,
      "variance": 2.211174985465456,
      "block_mean_std": 0.4750571147126127,
      "blocks": 8
    },
    "reference": {
      "mean": -3136.0119745813377,
      "std": 0.26786188351396384,
      "variance": 0.07174998863964832,
      "block_mean_std": 0.0071246754298508515,
      "blocks": 8
    },
    "mean_difference": 4.45916143568229,
    "wasserstein": 4.460067948215758
  },
  "energy_evaluation": "Unchanged raw BKS buck/coul-dsf energy; offline only",
  "total_energy_drift_meV_atom_ps": {
    "SMDP": 0.07756059165263471,
    "reference": 0.0005761529654454583
  },
  "energy_change_eV": {
    "SMDP": 10.712031757180284,
    "reference": 0.0047818902162362065
  },
  "nve_windows": [
    {
      "start_ps": 0,
      "end_ps": 1,
      "samples": 21,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 5.478471512767799,
          "change_eV": 0.05433933890383891,
          "mean_change_from_initial_eV": -0.28791208882157215
        },
        "PE": {
          "mean_eV": -3136.4128313629863,
          "change_eV": -1.3416738350979358,
          "mean_change_from_initial_eV": -0.627155500314313
        },
        "total": {
          "mean_eV": -3130.934359850218,
          "change_eV": -1.2873344961940347,
          "mean_change_from_initial_eV": -0.9150675891360152
        },
        "temperature_K": 263.25060848161405
      },
      "reference": {
        "KE": {
          "mean_eV": 5.949216012167396,
          "change_eV": 0.20226734978767524,
          "mean_change_from_initial_eV": 0.18283241057802385
        },
        "PE": {
          "mean_eV": -3135.970157144621,
          "change_eV": -0.20406274440847483,
          "mean_change_from_initial_eV": -0.18448128194937194
        },
        "total": {
          "mean_eV": -3130.020941132453,
          "change_eV": -0.0017953946207853733,
          "mean_change_from_initial_eV": -0.0016488713714958674
        },
        "temperature_K": 285.8707454335919
      }
    },
    {
      "start_ps": 1,
      "end_ps": 5,
      "samples": 80,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 8.118634077840738,
          "change_eV": 5.322392584219206,
          "mean_change_from_initial_eV": 2.352250476251368
        },
        "PE": {
          "mean_eV": -3133.8151962521633,
          "change_eV": 3.6156522414808023,
          "mean_change_from_initial_eV": 1.9704796105080504
        },
        "total": {
          "mean_eV": -3125.6965621743225,
          "change_eV": 8.938044825699762,
          "mean_change_from_initial_eV": 4.3227300867592815
        },
        "temperature_K": 390.11526409332043
      },
      "reference": {
        "KE": {
          "mean_eV": 5.966091527776227,
          "change_eV": 0.19501941432183578,
          "mean_change_from_initial_eV": 0.1997079261868557
        },
        "PE": {
          "mean_eV": -3135.988409114914,
          "change_eV": -0.1996197458001916,
          "mean_change_from_initial_eV": -0.20273325224258656
        },
        "total": {
          "mean_eV": -3130.0223175871374,
          "change_eV": -0.004600331478741282,
          "mean_change_from_initial_eV": -0.003025326055859523
        },
        "temperature_K": 286.6816449230047
      }
    },
    {
      "start_ps": 5,
      "end_ps": 20,
      "samples": 299,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 10.982004307425436,
          "change_eV": 0.9411462936941177,
          "mean_change_from_initial_eV": 5.215620705836066
        },
        "PE": {
          "mean_eV": -3131.1768104785842,
          "change_eV": 3.7773905493781967,
          "mean_change_from_initial_eV": 4.608865384087202
        },
        "total": {
          "mean_eV": -3120.1948061711582,
          "change_eV": 4.718536843072343,
          "mean_change_from_initial_eV": 9.824486089923148
        },
        "temperature_K": 527.7054575422754
      },
      "reference": {
        "KE": {
          "mean_eV": 6.014966852122372,
          "change_eV": 0.07261718468276257,
          "mean_change_from_initial_eV": 0.2485832505330013
        },
        "PE": {
          "mean_eV": -3136.037434548437,
          "change_eV": -0.07235549651295514,
          "mean_change_from_initial_eV": -0.2517586857659345
        },
        "total": {
          "mean_eV": -3130.022467696315,
          "change_eV": 0.0002616881702124374,
          "mean_change_from_initial_eV": -0.00317543523305629
        },
        "temperature_K": 289.0301939377934
      }
    },
    {
      "start_ps": 20,
      "end_ps": 100,
      "samples": 1600,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 10.50340338984075,
          "change_eV": 0.4698788775840619,
          "mean_change_from_initial_eV": 4.73701978825138
        },
        "PE": {
          "mean_eV": -3131.446171749636,
          "change_eV": -2.1270942929818375,
          "mean_change_from_initial_eV": 4.339504113035113
        },
        "total": {
          "mean_eV": -3120.9427683597955,
          "change_eV": -1.6572154153977863,
          "mean_change_from_initial_eV": 9.076523901286372
        },
        "temperature_K": 504.7078052809837
      },
      "reference": {
        "KE": {
          "mean_eV": 5.989119717947472,
          "change_eV": 0.21752980580055503,
          "mean_change_from_initial_eV": 0.22273611635810128
        },
        "PE": {
          "mean_eV": -3136.008943877163,
          "change_eV": -0.2066138776549451,
          "mean_change_from_initial_eV": -0.22326801449213235
        },
        "total": {
          "mean_eV": -3130.0198241592157,
          "change_eV": 0.010915928145550424,
          "mean_change_from_initial_eV": -0.0005318981341594053
        },
        "temperature_K": 287.7881917145108
      }
    }
  ],
  "thermostat": {
    "applicable": false
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.710039166659803,
          "block_overlap": {
            "mean": 0.7054079729963435,
            "std": 0.013767924520664694,
            "variance": 0.00018955574560672013,
            "block_mean_std": 0.013767924520664694,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9766100112697568
        },
        "14": {
          "overlap": 0.7964665698168524,
          "block_overlap": {
            "mean": 0.7909595277181115,
            "std": 0.014080662472733214,
            "variance": 0.00019826505567103745,
            "block_mean_std": 0.014080662472733214,
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
          "overlap": 0.7078621218337865,
          "block_overlap": {
            "mean": 0.7042839616954873,
            "std": 0.012080796660858508,
            "variance": 0.0001459456479610101,
            "block_mean_std": 0.012080796660858508,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9760548538076976
        },
        "14": {
          "overlap": 0.7862234425452813,
          "block_overlap": {
            "mean": 0.7819906092435156,
            "std": 0.014213703346480628,
            "variance": 0.0002020293628217546,
            "block_mean_std": 0.014213703346480628,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9782485236482704
        }
      }
    }
  },
  "task_status": {
    "thermo": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 0.21864020894281566
    },
    "rdf": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 2.0579017270356417
    },
    "energy": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 38.52060698706191
    },
    "thermostat": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 0.00039876997470855713
    },
    "vdos_macro": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 4.822236734093167
    },
    "vdos_dense": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 5.646901876898482
    }
  },
  "physical_analysis_completed": true,
  "analysis_failure_reason": {}
}
```

## nve_test_10: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition/nve_test_10.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/raw_nve/test_10/trajectory.h5",
  "production_available": true,
  "trajectory_completed": true,
  "stable": true,
  "failure": "",
  "evaluation_phase": "production",
  "duration_ps": 100.0,
  "requested_production_ps": 100.0,
  "temperature_K": {
    "SMDP": {
      "mean": 506.1891682319819,
      "std": 75.04242733223955,
      "variance": 5631.3658999144545,
      "block_mean_std": 24.88046988522437,
      "blocks": 8
    },
    "reference": {
      "mean": 302.7117153987662,
      "std": 13.544567609039127,
      "variance": 183.4553117158319,
      "block_mean_std": 0.47524526242154613,
      "blocks": 8
    },
    "mean_difference": 203.4774528332157,
    "wasserstein": 203.6374157202546
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 10.534231826568488,
      "std": 1.5616974363701623,
      "variance": 2.438898882765137,
      "block_mean_std": 0.5177839712381854,
      "blocks": 8
    },
    "reference": {
      "mean": 6.299691077481552,
      "std": 0.28187409794367413,
      "variance": 0.07945300709156,
      "block_mean_std": 0.009890262540214223,
      "blocks": 8
    },
    "mean_difference": 4.2345407490869365,
    "wasserstein": 4.2378697143067985
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 7.743993343501388,
        "std": 1.3092108871095465,
        "variance": 1.7140331469261654,
        "block_mean_std": 0.4100881925332335,
        "blocks": 8
      },
      "reference": {
        "mean": 4.2171512847601225,
        "std": 0.26756697380678635,
        "variance": 0.0715920854721215,
        "block_mean_std": 0.02948678633418141,
        "blocks": 8
      },
      "mean_difference": 3.5268420587412654,
      "wasserstein": 3.526842058741267
    },
    "14": {
      "SMDP": {
        "mean": 2.790238483067099,
        "std": 0.4298858569063612,
        "variance": 0.18480184996811647,
        "block_mean_std": 0.1362676162387017,
        "blocks": 8
      },
      "reference": {
        "mean": 2.082539792721431,
        "std": 0.21225893510662974,
        "variance": 0.045053855532600454,
        "block_mean_std": 0.029218059545168897,
        "blocks": 8
      },
      "mean_difference": 0.7076986903456683,
      "wasserstein": 0.7089658182451163
    }
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.0006742233832085103,
    "std": 5.0911005164595444e-05,
    "variance": 2.5919304468694637e-09,
    "block_mean_std": 1.487000816556545e-05,
    "blocks": 8
  },
  "minimum_distance_A": 1.4508488178253174,
  "thermodynamic_sampling": "2000 uniform production frames, matching R1",
  "RDF": {
    "14-8": {
      "RMSE": 0.8419542853692867,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 26.189443912141186,
        "reference": 34.6474984533892,
        "difference": -8.45805454124801
      },
      "block_curve_RMSE": {
        "mean": 0.846620980519539,
        "std": 0.10538750762892522,
        "variance": 0.011106526764236773,
        "block_mean_std": 0.10538750762892522,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.28337405306542107,
      "peak_position_A": {
        "SMDP": 2.624727889102222,
        "reference": 2.6009746955356867,
        "difference": 0.02375319356653538
      },
      "peak_height": {
        "SMDP": 7.331073839651932,
        "reference": 9.071304469266204,
        "difference": -1.7402306296142722
      },
      "block_curve_RMSE": {
        "mean": 0.28643661973478307,
        "std": 0.034264562523181184,
        "variance": 0.0011740602449049923,
        "block_mean_std": 0.034264562523181184,
        "blocks": 8
      }
    }
  },
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3131.416077488262,
      "std": 1.5475719642122554,
      "variance": 2.3949789844157783,
      "block_mean_std": 0.5111573141914226,
      "blocks": 8
    },
    "reference": {
      "mean": -3135.704982468285,
      "std": 0.2832842864853621,
      "variance": 0.08024998696952068,
      "block_mean_std": 0.011870085262351367,
      "blocks": 8
    },
    "mean_difference": 4.288904980022835,
    "wasserstein": 4.289784575788792
  },
  "energy_evaluation": "Unchanged raw BKS buck/coul-dsf energy; offline only",
  "total_energy_drift_meV_atom_ps": {
    "SMDP": 0.10208091171882433,
    "reference": -0.0004707604907260122
  },
  "energy_change_eV": {
    "SMDP": 8.570922720184171,
    "reference": 0.0032830016489242553
  },
  "nve_windows": [
    {
      "start_ps": 0,
      "end_ps": 1,
      "samples": 21,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 5.772886899689166,
          "change_eV": -0.27445768691076733,
          "mean_change_from_initial_eV": -0.2885731276745024
        },
        "PE": {
          "mean_eV": -3136.1589107113223,
          "change_eV": -0.9174981142405159,
          "mean_change_from_initial_eV": -0.6961667847365443
        },
        "total": {
          "mean_eV": -3130.3860238116335,
          "change_eV": -1.19195580115138,
          "mean_change_from_initial_eV": -0.9847399124112469
        },
        "temperature_K": 277.39780803768934
      },
      "reference": {
        "KE": {
          "mean_eV": 6.260652977180645,
          "change_eV": 0.3030401417840576,
          "mean_change_from_initial_eV": 0.1991929498169767
        },
        "PE": {
          "mean_eV": -3135.6615341435336,
          "change_eV": -0.3039474611696278,
          "mean_change_from_initial_eV": -0.19879021694785998
        },
        "total": {
          "mean_eV": -3129.4008811663525,
          "change_eV": -0.0009073193855329009,
          "mean_change_from_initial_eV": 0.0004027328689603781
        },
        "temperature_K": 300.8358630493965
      }
    },
    {
      "start_ps": 1,
      "end_ps": 5,
      "samples": 80,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 6.763590547200343,
          "change_eV": 3.5262068846410086,
          "mean_change_from_initial_eV": 0.702130519836675
        },
        "PE": {
          "mean_eV": -3135.1356651961023,
          "change_eV": 3.789618207305921,
          "mean_change_from_initial_eV": 0.3270787304831629
        },
        "total": {
          "mean_eV": -3128.372074648902,
          "change_eV": 7.315825091946863,
          "mean_change_from_initial_eV": 1.0292092503196215
        },
        "temperature_K": 325.002936114829
      },
      "reference": {
        "KE": {
          "mean_eV": 6.3416506282198,
          "change_eV": -0.09707309237203976,
          "mean_change_from_initial_eV": 0.28019060085613257
        },
        "PE": {
          "mean_eV": -3135.7430873307576,
          "change_eV": 0.09632933741204397,
          "mean_change_from_initial_eV": -0.2803434041724017
        },
        "total": {
          "mean_eV": -3129.4014367025384,
          "change_eV": -0.0007437549602400395,
          "mean_change_from_initial_eV": -0.00015280331647886668
        },
        "temperature_K": 304.7279488021371
      }
    },
    {
      "start_ps": 5,
      "end_ps": 20,
      "samples": 299,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 11.231904905399086,
          "change_eV": 0.36992993455386625,
          "mean_change_from_initial_eV": 5.170444878035419
        },
        "PE": {
          "mean_eV": -3130.716819716946,
          "change_eV": 0.82049551787577,
          "mean_change_from_initial_eV": 4.745924209639994
        },
        "total": {
          "mean_eV": -3119.4849148115463,
          "change_eV": 1.1904254524297357,
          "mean_change_from_initial_eV": 9.916369087675209
        },
        "temperature_K": 539.7136398104801
      },
      "reference": {
        "KE": {
          "mean_eV": 6.287557078350105,
          "change_eV": 0.030863639423835743,
          "mean_change_from_initial_eV": 0.22609705098643748
        },
        "PE": {
          "mean_eV": -3135.690439799746,
          "change_eV": -0.02730643160703039,
          "mean_change_from_initial_eV": -0.2276958731603734
        },
        "total": {
          "mean_eV": -3129.4028827213956,
          "change_eV": 0.0035572078168115695,
          "mean_change_from_initial_eV": -0.0015988221741226698
        },
        "temperature_K": 302.1286544761668
      }
    },
    {
      "start_ps": 20,
      "end_ps": 100,
      "samples": 1600,
      "endpoint_convention": "interpolated between identical sampled energy frames",
      "SMDP": {
        "KE": {
          "mean_eV": 10.654878886095721,
          "change_eV": 2.633324338251583,
          "mean_change_from_initial_eV": 4.5934188587320515
        },
        "PE": {
          "mean_eV": -3131.2985222128323,
          "change_eV": -1.3766963612924883,
          "mean_change_from_initial_eV": 4.1642217137526645
        },
        "total": {
          "mean_eV": -3120.643643326737,
          "change_eV": 1.2566279769589528,
          "mean_change_from_initial_eV": 8.757640572484513
        },
        "temperature_K": 511.98648081415786
      },
      "reference": {
        "KE": {
          "mean_eV": 6.300373016098779,
          "change_eV": 0.08570458701778882,
          "mean_change_from_initial_eV": 0.2389129887351122
        },
        "PE": {
          "mean_eV": -3135.7063651456074,
          "change_eV": -0.08432771884008616,
          "mean_change_from_initial_eV": -0.2436212190218069
        },
        "total": {
          "mean_eV": -3129.4059921295084,
          "change_eV": 0.0013768681778856262,
          "mean_change_from_initial_eV": -0.004708230286895798
        },
        "temperature_K": 302.74448380059385
      }
    }
  ],
  "thermostat": {
    "applicable": false
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.7094248007497681,
          "block_overlap": {
            "mean": 0.7020534719028744,
            "std": 0.02033208160709794,
            "variance": 0.00041339354247769034,
            "block_mean_std": 0.02033208160709794,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9739188229434805
        },
        "14": {
          "overlap": 0.7972498755738345,
          "block_overlap": {
            "mean": 0.7919225169396507,
            "std": 0.013593198339200205,
            "variance": 0.0001847750410888352,
            "block_mean_std": 0.013593198339200205,
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
          "overlap": 0.7119125928597857,
          "block_overlap": {
            "mean": 0.7038357931957792,
            "std": 0.019856079244954642,
            "variance": 0.0003942638829819185,
            "block_mean_std": 0.019856079244954642,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9729947939441181
        },
        "14": {
          "overlap": 0.7893666267621992,
          "block_overlap": {
            "mean": 0.7854023224398198,
            "std": 0.014960260972625193,
            "variance": 0.0002238094083690525,
            "block_mean_std": 0.014960260972625193,
            "blocks": 6
          },
          "reference_split_half_overlap": 0.9755400287991051
        }
      }
    }
  },
  "task_status": {
    "thermo": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 0.23634533293079585
    },
    "rdf": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 2.0938525249948725
    },
    "energy": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 38.476267814054154
    },
    "thermostat": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 0.00031027907971292734
    },
    "vdos_macro": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 4.775381849030964
    },
    "vdos_dense": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 5.878373033949174
    }
  },
  "physical_analysis_completed": true,
  "analysis_failure_reason": {}
}
```

## nvt: quantitative physical comparison

```json
{
  "candidate": "/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition/nvt.h5",
  "reference": "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/quartz/physical_reference/nvt_15ns/production.h5",
  "production_available": true,
  "trajectory_completed": true,
  "stable": true,
  "failure": "",
  "evaluation_phase": "production",
  "duration_ps": 15000.0,
  "requested_production_ps": 15000.0,
  "temperature_K": {
    "SMDP": {
      "mean": 319.4666842541695,
      "std": 26.000935721168528,
      "variance": 676.0486583763375,
      "block_mean_std": 1.728945321370098,
      "blocks": 8
    },
    "reference": {
      "mean": 299.1839998544485,
      "std": 18.658345123931916,
      "variance": 348.1338427637539,
      "block_mean_std": 1.0521074979856517,
      "blocks": 8
    },
    "mean_difference": 20.282684399721006,
    "wasserstein": 20.307267364716267
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 6.648376385755206,
      "std": 0.541101828692172,
      "variance": 0.29279118901401274,
      "block_mean_std": 0.03598083873344881,
      "blocks": 8
    },
    "reference": {
      "mean": 6.226276283775414,
      "std": 0.38829620499809836,
      "variance": 0.15077394281592524,
      "block_mean_std": 0.021895261664652284,
      "blocks": 8
    },
    "mean_difference": 0.4221001019797921,
    "wasserstein": 0.42261169461846615
  },
  "element_kinetic_energy_eV": {
    "8": {
      "SMDP": {
        "mean": 5.023428723361592,
        "std": 0.47086757410278796,
        "variance": 0.2217162723414445,
        "block_mean_std": 0.033941231001605926,
        "blocks": 8
      },
      "reference": {
        "mean": 4.1504345011315,
        "std": 0.3216096183131783,
        "variance": 0.10343274659154822,
        "block_mean_std": 0.020163058479425337,
        "blocks": 8
      },
      "mean_difference": 0.8729942222300924,
      "wasserstein": 0.8729942222300924
    },
    "14": {
      "SMDP": {
        "mean": 1.6249476623936139,
        "std": 0.24157586993566718,
        "variance": 0.05835890093517439,
        "block_mean_std": 0.009382727611829478,
        "blocks": 8
      },
      "reference": {
        "mean": 2.0758417826439146,
        "std": 0.22705827190739697,
        "variance": 0.051555458841573414,
        "block_mean_std": 0.010641413793087704,
        "blocks": 8
      },
      "mean_difference": -0.45089412025030073,
      "wasserstein": 0.45089412025030035
    }
  },
  "segment_join_derivative_RMS_A_fs": {
    "mean": 0.0005003085648258716,
    "std": 2.686559595731657e-05,
    "variance": 7.217602461417845e-10,
    "block_mean_std": 1.0780736366940808e-06,
    "blocks": 8
  },
  "minimum_distance_A": 1.4571613073349,
  "thermodynamic_sampling": "2000 uniform production frames, matching R1",
  "RDF": {
    "14-8": {
      "RMSE": 0.17638925863876875,
      "peak_position_A": {
        "SMDP": 1.6033405657411766,
        "reference": 1.6033405657411766,
        "difference": 0.0
      },
      "peak_height": {
        "SMDP": 36.21298971139139,
        "reference": 34.76711117175737,
        "difference": 1.4458785396340161
      },
      "block_curve_RMSE": {
        "mean": 0.19826245350018706,
        "std": 0.03506925192163563,
        "variance": 0.0012298524303431444,
        "block_mean_std": 0.03506925192163563,
        "blocks": 8
      }
    },
    "8-8": {
      "RMSE": 0.2006901944622742,
      "peak_position_A": {
        "SMDP": 2.624727889102222,
        "reference": 2.6009746955356867,
        "difference": 0.02375319356653538
      },
      "peak_height": {
        "SMDP": 8.185602922480454,
        "reference": 9.167464113664485,
        "difference": -0.9818611911840307
      },
      "block_curve_RMSE": {
        "mean": 0.20476370261246662,
        "std": 0.00973637366934787,
        "variance": 9.479697222917052e-05,
        "block_mean_std": 0.00973637366934787,
        "blocks": 8
      }
    }
  },
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3135.328127885682,
      "std": 0.5288544976009021,
      "variance": 0.27968707963270256,
      "block_mean_std": 0.024825211393323034,
      "blocks": 8
    },
    "reference": {
      "mean": -3135.7729840013926,
      "std": 0.4010645690089914,
      "variance": 0.16085278851436804,
      "block_mean_std": 0.027322190507312053,
      "blocks": 8
    },
    "mean_difference": 0.44485611571053596,
    "wasserstein": 0.4448561157105753
  },
  "energy_evaluation": "Unchanged raw BKS buck/coul-dsf energy; offline only",
  "total_energy_drift_meV_atom_ps": {
    "SMDP": -6.191673410005282e-06,
    "reference": -8.94806839869057e-06
  },
  "energy_change_eV": {
    "SMDP": -0.2576006361059626,
    "reference": -1.2183689951662018
  },
  "thermostat": {
    "target_K": 300,
    "tau_fs": 100.0,
    "dt_fs": 16,
    "production_steps": 937500,
    "post_temperature_K": {
      "mean": 320.2704991183919,
      "std": 25.410530047045775,
      "variance": 645.6950372718162,
      "block_mean_std": 0.6579315506263959,
      "blocks": 8
    },
    "pre_temperature_K": {
      "mean": 323.81863130779624,
      "std": 27.195534700787494,
      "variance": 739.5971076617367,
      "block_mean_std": 0.7686663787582665,
      "blocks": 8
    },
    "alpha": {
      "mean": 0.9948494985883077,
      "std": 0.017337579453097534,
      "variance": 0.00030059166129246974,
      "block_mean_std": 0.0001586688307819099,
      "blocks": 8
    },
    "network_delta_KE_eV": {
      "mean": 0.07383892308054615,
      "std": 0.8242067191162052,
      "variance": 0.6793167158362992,
      "block_mean_std": 0.0023604384949027003,
      "blocks": 8
    },
    "csvr_delta_KE_eV": {
      "mean": -0.0738396816451936,
      "std": 0.23345058936829927,
      "variance": 0.05449917767640629,
      "block_mean_std": 0.002360044707287073,
      "blocks": 8
    },
    "post_step_delta_KE_eV": {
      "mean": -7.58564647412457e-07,
      "std": 0.7918713240378411,
      "variance": 0.6270601938334435,
      "block_mean_std": 4.86794375383249e-06,
      "blocks": 8
    },
    "scaling_max_error_K": 7.426293842627274e-05,
    "conditional_mean_innovation_K": {
      "mean": -0.026399612222768595,
      "std": 10.465002013285103,
      "variance": 109.51626713806125,
      "block_mean_std": 0.022768636533444696,
      "blocks": 8
    },
    "predicted_stationary_post_temperature_K": 320.44883826990065,
    "species": {
      "8": {
        "network_delta_KE_eV": {
          "mean": 0.056062106443278,
          "std": 0.6834052595741441,
          "variance": 0.4670427488136033,
          "block_mean_std": 0.0018234351963737985,
          "blocks": 8
        },
        "csvr_delta_KE_eV": {
          "mean": -0.05606253723398844,
          "std": 0.17671391817166307,
          "variance": 0.031227808875581235,
          "block_mean_std": 0.0018231670174496662,
          "blocks": 8
        },
        "pre_KE_eV": {
          "mean": 5.096913634987132,
          "std": 0.49282247655622896,
          "variance": 0.24287399339901486,
          "block_mean_std": 0.017587020682470388,
          "blocks": 8
        },
        "post_KE_eV": {
          "mean": 5.040851097753143,
          "std": 0.46525227530200136,
          "variance": 0.21645967967368926,
          "block_mean_std": 0.015818995239765765,
          "blocks": 8
        }
      },
      "14": {
        "network_delta_KE_eV": {
          "mean": 0.01777681646874746,
          "std": 0.33289462231912714,
          "variance": 0.1108188295689943,
          "block_mean_std": 0.0005410752233070703,
          "blocks": 8
        },
        "csvr_delta_KE_eV": {
          "mean": -0.01777714424279531,
          "std": 0.05741990881869453,
          "variance": 0.0032970459287471935,
          "block_mean_std": 0.0005406913780977563,
          "blocks": 8
        },
        "pre_KE_eV": {
          "mean": 1.642030509742991,
          "std": 0.24368259725172006,
          "variance": 0.059381208203344006,
          "block_mean_std": 0.004223130069617887,
          "blocks": 8
        },
        "post_KE_eV": {
          "mean": 1.6242533655001958,
          "std": 0.23805949260549378,
          "variance": 0.05667232201958514,
          "block_mean_std": 0.0043738860804891306,
          "blocks": 8
        }
      }
    }
  },
  "VDOS": {
    "macro_16fs": {
      "nyquist_THz": 31.25,
      "elements": {
        "8": {
          "overlap": 0.5991716137758726,
          "block_overlap": {
            "mean": 0.5958568068589138,
            "std": 0.019198222402942916,
            "variance": 0.0003685717434328593,
            "block_mean_std": 0.0032102725028941874,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9987517641216376
        },
        "14": {
          "overlap": 0.7313264882776215,
          "block_overlap": {
            "mean": 0.7244186104743512,
            "std": 0.017113179797677392,
            "variance": 0.00029286092278763367,
            "block_mean_std": 0.002797809828891333,
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
          "overlap": 0.6070198437881021,
          "block_overlap": {
            "mean": 0.603208262394555,
            "std": 0.018496845815989923,
            "variance": 0.0003421333051405039,
            "block_mean_std": 0.002974591879450844,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9986130057655248
        },
        "14": {
          "overlap": 0.7280543236715831,
          "block_overlap": {
            "mean": 0.7204657328217495,
            "std": 0.017128557270186175,
            "variance": 0.00029338747415804767,
            "block_mean_std": 0.002761253504307661,
            "blocks": 8
          },
          "reference_split_half_overlap": 0.9984999858738758
        }
      }
    }
  },
  "task_status": {
    "thermo": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 56.96368976100348
    },
    "rdf": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 23.776707620942034
    },
    "energy": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 120.51865372795146
    },
    "thermostat": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 28.509240690036677
    },
    "vdos_macro": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 457.7362090059323
    },
    "vdos_dense": {
      "completed": true,
      "failure_reason": null,
      "elapsed_s": 693.8409150199732
    }
  },
  "physical_analysis_completed": true,
  "analysis_failure_reason": {}
}
```

## Observed mechanism and cost limitations

Additive mode readout; no competitive fusion weights. Report actual per-mode displacement/velocity RMS and signed projections from the selected checkpoint test. These attributions are not named physical modes.
Soft routing evaluates both recurrent and temporal operators. Inference cost includes all 16 time queries; a learned route does not save expert computation. No matched-hardware BKS speedup is claimed without a measured baseline.

## Interpretation boundaries

Only completed production frames are production evidence. Failed thermalization statistics are diagnostics, not 15 ns NVT results.
The stable field only means no numerical termination threshold was triggered. It is not evidence of physical fidelity or energy conservation.
16 fs spectra compare identically downsampled outputs. Queried 1 fs spectra compare dense velocities separately and include endpoint thermostat kicks; output cadence alone does not establish high-frequency fidelity.
This experiment stops after these evaluations; there is no automatic architecture change or second training run.
