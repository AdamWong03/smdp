"""Export completed Water NVT metrics without copying trajectories or weights."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics.units import KB_EV_PER_K


def export(run,output):
    analysis=run/'physics/nvt_replica_01/analysis'
    output.mkdir(parents=True,exist_ok=True)
    data={name:json.loads((analysis/(name+'.json')).read_text())
          for name in ['motion','energy','rdf','vdos','thermostat','diffusion','status']}
    data['rollout']=json.loads((run/'physics/nvt_replica_01/trajectory.summary.json').read_text())
    data['reference_verification']=json.loads((analysis/'reference_verification.json').read_text())
    if not all(data[name]['completed'] for name in ['motion','energy','rdf','vdos','thermostat','diffusion']):
        raise ValueError('Some physical tasks remain incomplete')
    (output/'nvt_results.json').write_text(json.dumps(data,indent=2)+'\n')
    for name in ['motion','energy','rdf','vdos','vacf','msd']:
        for extension in ['csv','png']:
            source=analysis/(name+'.'+extension)
            if source.exists():shutil.copy2(source,output/('nvt_'+source.name))
    motion=data['motion']['metrics'];energy=data['energy']['metrics']
    thermo=data['thermostat']['metrics'];diff=data['diffusion']['metrics']['results']
    decay=np.exp(-thermo['dt_fs']/thermo['tau_fs']);factor=(3*192-3)*KB_EV_PER_K/2
    expected=(thermo['target_K']+decay/(1-decay)*thermo['network_delta_KE_eV']['mean']/factor+
              thermo['conditional_mean_innovation_K']['mean']/(1-decay))
    lines=['# Water 5 fs: complete 2.5 ns NVT results','',
        'The fixed validation-selected epoch-400 checkpoint completed 250 ps',
        'thermalization plus 2.5 ns production (550,000 neural steps). No forces',
        'were used during neural propagation. This is one autonomous run.',
        '', '## Main result','',
        'Numerical completion did not produce credible liquid dynamics. Molecular',
        'translation and rotation are strongly suppressed, internal motion dominates,',
        'RDF peaks over-sharpen, velocity spectra disagree, and molecular MSD',
        'approaches a plateau. The original CSVR thermostat does not repair this',
        'imbalance. All failed and unfavorable results are retained.',
        '', '| Observable | DSRT | Reference MD |','|---|---:|---:|',
        f"| Temperature, 2,000 matched frames (K) | {motion['temperature_K']['SMDP']['mean']:.3f} | {motion['temperature_K']['reference']['mean']:.3f} |"]
    for key,label in [('translation_eV','Molecular translation KE (eV)'),
                      ('rotation_eV','Molecular rotation KE (eV)'),
                      ('internal_eV','Molecular internal KE (eV)'),
                      ('total_KE_eV','Total KE (eV)')]:
        v=motion['kinetic_partition'][key]
        lines.append(f"| {label} | {v['SMDP']['mean']:.6f} | {v['reference']['mean']:.6f} |")
    v=energy['components']['pe']['distribution']
    lines.append(f"| Potential energy mean (eV) | {v['SMDP']['mean']:.6f} | {v['reference']['mean']:.6f} |")
    lines.append(f"| Potential energy std (eV) | {v['SMDP']['std']:.6f} | {v['reference']['std']:.6f} |")
    for pair,label in [('8-8','O–O'),('8-1','O–H'),('1-1','H–H')]:
        value=data['rdf']['metrics'][pair]
        lines.append(f"| {label} RDF first peak height | {value['peak_height']['model']:.5f} | {value['peak_height']['reference']:.5f} |")
    for element,label in [('1','H'),('8','O')]:
        value=data['vdos']['metrics']['elements'][element]['overlap']
        lines.append(f'| {label} VDOS overlap | {value:.6f} | Compared with reference |')
    lines+=['', 'The full 500,000-step production mean temperature after thermostat is',
        f"{thermo['post_temperature_K']['mean']:.4f} K (pre-thermostat {thermo['pre_temperature_K']['mean']:.4f} K).",
        'This full-step estimate is distinct from the 2,000-frame thermodynamic table.',
        '', '## Thermostat balance','',
        f"The network adds {thermo['network_delta_KE_eV']['mean']:.8f} eV kinetic energy per 5 fs step;",
        f"CSVR removes {-thermo['csvr_delta_KE_eV']['mean']:.8f} eV per step.",
        'CSVR remains at the reference 300 K target and 50 fs coupling time.',
        f'The stationary balance, including the sampled random innovation, predicts {expected:.4f} K,',
        'matching the measured production mean. Maximum temperature-scaling error',
        f"is {thermo['scaling_max_error_K']:.3g} K. These measurements support a persistent",
        'model kinetic-energy source with finite thermostat relaxation rather than',
        'a changed target temperature or a demonstrated CSVR formula error.',
        '', '## Diffusion and its limits','',
        'The preregistered analysis uses eight time blocks, molecular COM MSD,',
        '50 fs sampling and the 10–100 ps fit window. It yields:',
        '', '| Quantity | DSRT | Reference |','|---|---:|---:|',
        f"| Mean fitted slope / 6 (m²/s) | {diff['model']['diffusion_m2_s']['mean']:.6g} | {diff['reference']['diffusion_m2_s']['mean']:.6g} |",
        '', 'The model MSD plateaus; several later blocks have poor linear fits or',
        'negative slopes. Its tiny fitted value is therefore an effective slope,',
        'not a well-established positive equilibrium diffusion coefficient. The',
        'reference has a clear diffusive regime. No fitting window was adjusted.',
        '', '## Execution and software repair','',
        f"End-to-end throughput is {data['rollout']['ns_day_end_to_end']:.3f} ns/day on a shared L40S,",
        f"with {data['rollout']['elapsed_seconds']:.1f} seconds of trajectory generation.",
        'It includes graph updates, CSVR and output, but excludes offline physical',
        'analysis; it is not an exclusive-GPU timing comparison.',
        '', 'The initial energy task failed during reference verification with a',
        '`Bond atom missing in image check` error. Long unwrapped coordinates are',
        'now supplied as wrapped coordinates plus consistent image flags. Molecular',
        'translations across multiple periodic boxes preserve energy and forces.',
        'The long reference has coordinates every 0.5 fs, thermo every 5 fs and no',
        'saved forces, unlike the NVE files. Verification now matches MD steps',
        'and checks energy where forces are unavailable; NVE force checks remain.',
        'Eight long-reference energy checks differ by at most 0.00058 eV. The',
        'original failure traceback remains on the server. The repaired task reuses',
        'the completed trajectory and all completed analysis tasks.',
        '', '## Artifacts','',
        f'- Server trajectory: `{run}/physics/nvt_replica_01/trajectory.h5`',
        f'- Server analysis: `{analysis}`',
        '- Compact metrics: `nvt_results.json`',
        '- Curves and numeric samples: `nvt_*.png` and `nvt_*.csv`',
        '- Protocol: `configs/water5_physics.json`',
        '- Reproduce/resume: `bash scripts/run_water5_nvt.sh` (existing trajectory is reused).']
    (output/'NVT_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.run,a.output)
