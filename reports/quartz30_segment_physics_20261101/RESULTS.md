# SMDP / TrajCast infrastructure / Quartz 30 fs

All SMDP parameters start from random initialization. No pretrained weights or additional baseline training.
Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_segment_physics_20261101`.
Method and commands: [README](../../trajcast_smdp/README.md).

Training started: False.
Training completed: False.
Independent test evaluation completed: True.

Execution snapshot: 2026-11-01T15:14:02.530348+00:00; service {'MainPID': '3326217', 'ActiveState': 'active', 'SubState': 'running'}.
Free disk at snapshot: 16.00 GiB; trajectory storage is checked against requested duration before launch.
See progress_snapshot.json for the last actual training record. Early training loss is not a physical result.

## Frozen segment checkpoint evaluation

Source: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028/segment/best.pt`; SHA256 `36c0789a3919211fc14df643c7daa2c4a82940e29dd9ecca1ac81a5244b572dd`.
This run performs no training. It uses the segment-only validation-selected checkpoint, not rollout-trained weights.
Requested: two 100 ps NVE runs; NVT 1.5 ns thermalization + 1 ns production. This is not a 15 ns result.
NVT production length is limited by available storage. Dynamics, thermostat and analysis conventions are unchanged.
30 fs spectra have a 16.67 THz Nyquist limit; 1 fs queried spectra are evaluated separately.
BKS absolute energies use the same raw DSF convention for predicted and reference configurations; no centering.

Endpoint position RMSE: 0.001894814 Å.
Endpoint position MAE: 0.001502626 Å.
Endpoint velocity RMSE: 0.0001557011 Å/fs.
Endpoint velocity MAE: 0.0001232798 Å/fs.
Eight-step normalized error: 0.04146408.

## nve_0

Trajectory completed: True; duration 99.99 ps (includes any thermalization).
End-to-end throughput: 22.28 ns/day.
Failure: None.

### thermo

```json
{
  "temperature_K": {
    "SMDP": {
      "mean": 262.1652080156821,
      "std": 48.568281898466395,
      "variance": 2358.8780065688984,
      "block_mean_std": 18.670169554213054,
      "blocks": 8
    },
    "reference": {
      "mean": 288.40293907938536,
      "std": 13.211470955188958,
      "variance": 174.54296479980144,
      "block_mean_std": 1.2089529408179962,
      "blocks": 8
    },
    "mean_difference": -26.23773106370328,
    "wasserstein": 36.03536303733563
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 5.455883395813953,
      "std": 1.0107477066796886,
      "variance": 1.02161092655825,
      "block_mean_std": 0.38854228155922604,
      "blocks": 8
    },
    "reference": {
      "mean": 6.00191313918756,
      "std": 0.2749420701711956,
      "variance": 0.07559314195002265,
      "block_mean_std": 0.025159350190108953,
      "blocks": 8
    },
    "mean_difference": -0.546029743373607,
    "wasserstein": 0.7499268890239889
  },
  "minimum_distance_A": 1.4824389510129816,
  "per_element": {
    "8": {
      "SMDP": {
        "mean": 3.5797555172023072,
        "std": 0.6542016327589195,
        "variance": 0.4279797763044362,
        "block_mean_std": 0.27173570742718617,
        "blocks": 8
      },
      "reference": {
        "mean": 3.9996176005514523,
        "std": 0.26158827652969935,
        "variance": 0.06842842641777847,
        "block_mean_std": 0.027278775329111785,
        "blocks": 8
      },
      "mean_difference": -0.4198620833491451,
      "wasserstein": 0.4777057573225686
    },
    "14": {
      "SMDP": {
        "mean": 1.8761278786116455,
        "std": 0.4290414913298044,
        "variance": 0.1840766012825026,
        "block_mean_std": 0.1278043461599399,
        "blocks": 8
      },
      "reference": {
        "mean": 2.002295538636107,
        "std": 0.1980465007058603,
        "variance": 0.03922241644183633,
        "block_mean_std": 0.019397423336111462,
        "blocks": 8
      },
      "mean_difference": -0.12616766002446145,
      "wasserstein": 0.21778043027648022
    }
  }
}
```

### energy

```json
{
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3136.588564600733,
      "std": 1.047962291450452,
      "variance": 1.0982249643020823,
      "block_mean_std": 0.39518753240552706,
      "blocks": 8
    },
    "reference": {
      "mean": -3136.022398475807,
      "std": 0.2762054886776075,
      "variance": 0.07628947197563597,
      "block_mean_std": 0.024646887001580447,
      "blocks": 8
    },
    "mean_difference": -0.5661661249259851,
    "wasserstein": 0.7812939139519646
  },
  "energy_convention": "raw BKS DSF, same parameters/units as own MD",
  "energy_change_eV": {
    "SMDP": -0.8636487275362015,
    "reference": 0.007815085255970189
  },
  "drift_meV_atom_ps": {
    "SMDP": 0.01818853030395251,
    "reference": 0.0005536380760884419
  }
}
```

### macro_30fs

```json
{
  "dt_fs": 30,
  "nyquist_THz": 16.666666666666668,
  "elements": {
    "8": {
      "overlap": 0.9041857321481788,
      "block_overlap": {
        "mean": 0.9000440929863417,
        "std": 0.015249310524192164,
        "variance": 0.00023254147146323787,
        "block_mean_std": 0.015249310524192164,
        "blocks": 6
      }
    },
    "14": {
      "overlap": 0.8532273007659474,
      "block_overlap": {
        "mean": 0.8512174692930419,
        "std": 0.020779314666166317,
        "variance": 0.0004317799179955546,
        "block_mean_std": 0.020779314666166317,
        "blocks": 6
      }
    }
  }
}
```

### dense_1fs

```json
{
  "dt_fs": 1,
  "nyquist_THz": 500.0,
  "elements": {
    "8": {
      "overlap": 0.8840144940782126,
      "block_overlap": {
        "mean": 0.8775355766533988,
        "std": 0.010342717703028659,
        "variance": 0.00010697180948454242,
        "block_mean_std": 0.010342717703028659,
        "blocks": 6
      }
    },
    "14": {
      "overlap": 0.8301431921346081,
      "block_overlap": {
        "mean": 0.8261673772361698,
        "std": 0.02161146521050636,
        "variance": 0.00046705542854492667,
        "block_mean_std": 0.02161146521050636,
        "blocks": 6
      }
    }
  }
}
```

### rdf

```json
{
  "14-8": {
    "RMSE": 0.16277908420579992,
    "peak_position_A": {
      "SMDP": 1.6033405657411766,
      "reference": 1.6033405657411766
    },
    "peak_height": {
      "SMDP": 37.358401102484734,
      "reference": 35.74984926586994
    }
  },
  "8-8": {
    "RMSE": 0.08032969544473999,
    "peak_position_A": {
      "SMDP": 2.6009746955356867,
      "reference": 2.6009746955356867
    },
    "peak_height": {
      "SMDP": 9.889946022053817,
      "reference": 9.232059905321348
    }
  }
}
```

## nve_1

Trajectory completed: True; duration 99.99 ps (includes any thermalization).
End-to-end throughput: 22.4 ns/day.
Failure: None.

### thermo

```json
{
  "temperature_K": {
    "SMDP": {
      "mean": 261.86028262211323,
      "std": 36.67290064647685,
      "variance": 1344.901641826362,
      "block_mean_std": 18.848395734751204,
      "blocks": 8
    },
    "reference": {
      "mean": 303.1415476462189,
      "std": 13.588414696060214,
      "variance": 184.64501395210522,
      "block_mean_std": 0.888523313995721,
      "blocks": 8
    },
    "mean_difference": -41.281265024105664,
    "wasserstein": 42.63524583115615
  },
  "kinetic_energy_eV": {
    "SMDP": {
      "mean": 5.449537636190371,
      "std": 0.7631945948429564,
      "variance": 0.5824659895975043,
      "block_mean_std": 0.3922513216200998,
      "blocks": 8
    },
    "reference": {
      "mean": 6.30863625613288,
      "std": 0.28278659352553964,
      "variance": 0.07996825747777879,
      "block_mean_std": 0.018490934141546424,
      "blocks": 8
    },
    "mean_difference": -0.8590986199425092,
    "wasserstein": 0.8872761247279488
  },
  "minimum_distance_A": 1.4892433116349162,
  "per_element": {
    "8": {
      "SMDP": {
        "mean": 3.532903519243804,
        "std": 0.48627378068990473,
        "variance": 0.23646218978645356,
        "block_mean_std": 0.22981770100056195,
        "blocks": 8
      },
      "reference": {
        "mean": 4.223022582320726,
        "std": 0.27224803410260695,
        "variance": 0.07411899207273423,
        "block_mean_std": 0.038462269888584644,
        "blocks": 8
      },
      "mean_difference": -0.6901190630769221,
      "wasserstein": 0.6929451770362266
    },
    "14": {
      "SMDP": {
        "mean": 1.9166341169465664,
        "std": 0.3598714907958494,
        "variance": 0.12950748988762711,
        "block_mean_std": 0.1727267162818886,
        "blocks": 8
      },
      "reference": {
        "mean": 2.085613673812154,
        "std": 0.21081542029634515,
        "variance": 0.04444314143472465,
        "block_mean_std": 0.029753007222714012,
        "blocks": 8
      },
      "mean_difference": -0.16897955686558763,
      "wasserstein": 0.19153699567277954
    }
  }
}
```

### energy

```json
{
  "potential_energy_eV": {
    "SMDP": {
      "mean": -3136.5946956921985,
      "std": 0.7764718905489364,
      "variance": 0.6029085968126394,
      "block_mean_std": 0.4147784150119534,
      "blocks": 8
    },
    "reference": {
      "mean": -3135.7140455769218,
      "std": 0.28413218461123263,
      "variance": 0.08073109833195159,
      "block_mean_std": 0.019449037765867046,
      "blocks": 8
    },
    "mean_difference": -0.8806501152766941,
    "wasserstein": 0.9129593425262716
  },
  "energy_convention": "raw BKS DSF, same parameters/units as own MD",
  "energy_change_eV": {
    "SMDP": -1.5610895380614238,
    "reference": 0.0019395575977796398
  },
  "drift_meV_atom_ps": {
    "SMDP": 0.04560879289003397,
    "reference": -0.0004723316865160373
  }
}
```

### macro_30fs

```json
{
  "dt_fs": 30,
  "nyquist_THz": 16.666666666666668,
  "elements": {
    "8": {
      "overlap": 0.8986408105079806,
      "block_overlap": {
        "mean": 0.891488650202736,
        "std": 0.014126126607030422,
        "variance": 0.00019954745291785283,
        "block_mean_std": 0.014126126607030422,
        "blocks": 6
      }
    },
    "14": {
      "overlap": 0.8369503403481827,
      "block_overlap": {
        "mean": 0.83351973127652,
        "std": 0.031715609788466076,
        "variance": 0.0010058799042542454,
        "block_mean_std": 0.031715609788466076,
        "blocks": 6
      }
    }
  }
}
```

### dense_1fs

```json
{
  "dt_fs": 1,
  "nyquist_THz": 500.0,
  "elements": {
    "8": {
      "overlap": 0.8773343257806502,
      "block_overlap": {
        "mean": 0.8701082549963539,
        "std": 0.010983337121851813,
        "variance": 0.00012063369433224805,
        "block_mean_std": 0.010983337121851813,
        "blocks": 6
      }
    },
    "14": {
      "overlap": 0.813269158293816,
      "block_overlap": {
        "mean": 0.8079776912759941,
        "std": 0.029027650807092645,
        "variance": 0.0008426045113785062,
        "block_mean_std": 0.029027650807092645,
        "blocks": 6
      }
    }
  }
}
```

### rdf

```json
{
  "14-8": {
    "RMSE": 0.26003844117981895,
    "peak_position_A": {
      "SMDP": 1.6033405657411766,
      "reference": 1.6033405657411766
    },
    "peak_height": {
      "SMDP": 37.37753913742365,
      "reference": 34.60156716953583
    }
  },
  "8-8": {
    "RMSE": 0.10697350357099757,
    "peak_position_A": {
      "SMDP": 2.6009746955356867,
      "reference": 2.6009746955356867
    },
    "peak_height": {
      "SMDP": 9.815257137950573,
      "reference": 9.054788499808485
    }
  }
}
```

## nvt_0

Not completed; no physical results claimed.

No phonon identification, stability improvement or speedup is inferred from architecture alone.
Official-paper BKS energy discrepancies remain recorded in the foundation report. This experiment compares only against its own independently generated, unchanged BKS reference.
