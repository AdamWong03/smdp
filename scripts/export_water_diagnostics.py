"""Publish compact, measured Water physics/bias results; keep large data local."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(run, out):
    out.mkdir(parents=True,exist_ok=True)
    analysis=run/'physics/nve_test_05/analysis'
    bias=run/'evaluation/signed_bias_deployed'
    if not (bias/'results.json').exists():
        bias=run/'evaluation/signed_bias'
    metrics={}
    for name in ['motion','energy','rdf','vdos','full_energy','reference_verification','status']:
        source=analysis/(name+'.json')
        if source.exists():
            metrics[name]=json.loads(source.read_text())
    metrics['signed_bias']=json.loads((bias/'results.json').read_text())
    metrics['diagnostic_reference_verification']=json.loads((bias/'reference_verification.json').read_text())
    (out/'physics_results.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name in ['motion.png','energy.png','rdf.png','vdos.png','full_candidate_energy.png',
                 'motion.csv','energy.csv','full_candidate_energy.csv','rdf.csv','vdos.csv']:
        if (analysis/name).exists():shutil.copy2(analysis/name,out/name)
    shutil.copy2(bias/'signed_errors.csv',out/'signed_errors.csv')
    groups=metrics['signed_bias']['groups'];labels=list(groups)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(labels))
    for ax,keys in zip(axes,[['kinetic_error_eV','potential_error_eV','total_energy_error_eV'],
                            ['Z1_kinetic_error_eV','Z8_kinetic_error_eV']]):
        for key in keys:ax.plot(x,[groups[k][key]['mean'] for k in labels],'o-',label=key)
        ax.axhline(0,color='grey',linewidth=.8);ax.set_xticks(x)
        ax.set_xticklabels([k.replace('reference/','real ').replace('autonomous/','') for k in labels],rotation=30)
        ax.set_ylabel('Mean signed error (eV/5 fs step)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'signed_bias.png',dpi=180);plt.close(fig)

    motion=metrics['motion']['metrics'];energy=metrics['energy']['metrics']
    lines=['# Water: energy, motion and signed prediction-error diagnosis','',
        'Checkpoint: validation-selected epoch 400; no retraining or architecture change.',
        'NVE comparison uses 2,000 matched frames over 0–50 ps. The model-only',
        'trajectory continues to 100 ps. Frames are correlated; this is one run.',
        '', '## Reference dynamics verification','',
        'The offline calculator uses the original flexible SPC/Fw topology, bonded',
        'terms, excluded intramolecular pairs and PPPM 1e-5. Raw energies were not',
        'centered. Eight fixed reference frames were checked against saved forces',
        'and potential energy. The maximum differences are about 0.00032 eV and',
        '0.0056 eV/Å; rounded raw dump coordinates limit reproducibility.',
        'A fresh 5 fs continuation reproduces the stored endpoint within',
        '0.000106 Å and 0.0000215 Å/fs (maximum component errors).',
        '', '## NVE energy and motion','',
        '| Quantity | Model | Reference |','|---|---:|---:|']
    for key,label in [('translation_eV','Mean molecular translation KE (eV)'),
                      ('rotation_eV','Mean molecular rotation KE (eV)'),
                      ('internal_eV','Mean molecular internal KE (eV)')]:
        v=motion['kinetic_partition'][key]
        lines.append(f"| {label} | {v['SMDP']['mean']:.5f} | {v['reference']['mean']:.5f} |")
    for key,label in [('pe','0–50 ps potential energy change (eV)'),('ke','0–50 ps kinetic energy change (eV)'),
                      ('total','0–50 ps total energy change (eV)'),('ebond','0–50 ps bond energy change (eV)'),
                      ('nonbonded','0–50 ps nonbonded energy change (eV)')]:
        v=energy['components'][key]['change_eV']
        lines.append(f"| {label} | {v['model']:.5f} | {v['reference']:.5f} |")
    lines+=['', 'Internal kinetic energy is strongly overexcited while molecular translation',
        'and rotation are underexcited. O–H bond-length standard deviation is',
        f"{motion['bond_lengths_A']['SMDP']['std']:.5f} Å versus {motion['bond_lengths_A']['reference']['std']:.5f} Å.",
        'The angle distribution is narrower, not uniformly hotter. Total energy',
        'increases, so the temperature rise is not explained by redistribution alone.',
        '', '## Signed endpoint bias','',
        'The existing 1,000 test pairs provide real-state labels. For autonomous',
        'states, 32 states were sampled in each of five time windows and the original',
        'SPC/Fw potential generated an independent 5 fs continuation from each state',
        'itself (160 diagnostic labels, 0.8 ps total reference integration). These',
        'labels are never training data. No sample failed or was filtered out.',
        '', '| Input states | Samples | Mean total-energy endpoint error (eV) | Std (eV) | Mean velocity RMSE (Å/fs) |',
        '|---|---:|---:|---:|---:|']
    for k,v in groups.items():
        lines.append(f"| {k} | {v['samples']} | {v['total_energy_error_eV']['mean']:+.6f} | {v['total_energy_error_eV']['std']:.6f} | {v['velocity_RMSE_A_fs']['mean']:.7f} |")
    lines+=['', 'The real-state test has a small negative mean energy bias. Early autonomous',
        'states have positive sample means; later intervals have larger variance and',
        'changing signs. This contradicts a simple constant velocity-amplification',
        'explanation but does not identify a unique causal network mechanism.',
        'Autonomous inputs also have different temperatures and kinetic partition,',
        'so temperature and state-distribution effects remain confounded.',
        'Stratified sample means are diagnostics, not precise long-time energy rates.',
        'Signed per-element errors are conditional on the current state and must not',
        'be confused with accumulated per-element energy: H can already be too hot',
        'while its next-step conditional kinetic-energy error is negative.',
        '', '## Structure and velocity spectra','',
        '| Metric | Value |','|---|---:|']
    for key,v in metrics['rdf']['metrics'].items():lines.append(f"| {key} RDF RMSE | {v['RMSE']:.6f} |")
    for key,v in metrics['vdos']['metrics']['elements'].items():lines.append(f"| Z={key} VDOS overlap | {v['overlap']:.6f} |")
    lines+=['', 'The O–O peak is substantially over-sharpened. VDOS uses 5 fs matched',
        'sampling (100 THz Nyquist); the heated NVE is nonstationary, so these',
        'overlaps are not equilibrium spectra or an independent diffusion result.',
        '', '## Follow-up NVT protocol','',
        'One 300 K CSVR run uses the original 50 fs coupling time, 250 ps',
        'thermalization and 2.5 ns production. The frozen checkpoint is unchanged.',
        'Per-element pre/post thermostat kinetic energies are recorded. Molecular',
        'COM diffusion uses eight blocks, 50 fs sampling and a predeclared 10–100 ps',
        'MSD fitting interval. The GPU is shared, so runtime is not an exclusive-GPU',
        'benchmark. Completed NVT results are reported separately in NVT_RESULTS.md.',
        '', '## Software correction','',
        'A new diagnostic initially failed an algebraic kinetic-error identity due',
        'to mixing float32 squaring with float64 evaluation (1.5e-9 eV discrepancy).',
        'All offline kinetic arithmetic now uses float64; the identity passes.',
        'No model weights, labels or trajectory were changed by this correction.',
        '', '## Reproduction','', '```bash',
        'LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib OMP_NUM_THREADS=1 \\',
        '/data1/wyd/envs/mlip/bin/python analyze_water.py \\',
        f'  --candidate {run}/physics/nve_test_05/trajectory.h5 \\',
        '  --reference /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/raw_nve/test_05/trajectory.h5 \\',
        '  --topology /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data \\',
        f'  --output {analysis}', '```', '',
        'See `configs/water5_physics.json`, `diagnose_water_bias.py` and',
        '`scripts/run_water5_nvt.sh` for the remaining exact execution settings.',
        'Large trajectory and diagnostic-label HDF5 files stay on the server.']
    if 'full_energy' in metrics:
        value=metrics['full_energy']['metrics']['components']['total']['change_eV']
        lines.insert(lines.index('## Signed endpoint bias'),
                     f'The complete model-only 100 ps total-energy change is {value:+.5f} eV.\n')
    (out/'PHYSICS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    export(args.run,args.output)
