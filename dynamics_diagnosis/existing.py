"""Re-use completed production statistics without touching active trajectories."""
import json

import pandas as pd

from dynamics_diagnosis.common import REPORT,checkpoints


def main():
    rows=[]
    for system in ['quartz','water']:
        for model,checkpoint in checkpoints(system).items():
            for summary in sorted((checkpoint.parent/'physics').glob('*/trajectory.summary.json')):
                if 'validation' in summary.parent.name:continue
                generation=json.loads(summary.read_text())
                analysis=summary.parent/'analysis'
                primary=analysis/('metrics.json' if system=='quartz' else 'status.json')
                if not primary.exists():continue
                status=json.loads(primary.read_text())
                if not status.get('physical_analysis_completed'):continue
                row=dict(system=system,model=model,case=summary.parent.name,
                    generation_completed=bool(generation['stable']),
                    simulated_ps=generation['simulated_ps'],ns_day=generation['ns_day_end_to_end'],
                    source_metrics=str(primary),same_initial_conditions='test production protocol, separate from validation diagnosis')
                if system=='quartz':
                    metrics=status
                    for key in ['temperature_K','potential_energy_eV','kinetic_energy_eV']:
                        row[key+'_mean']=metrics[key]['SMDP']['mean']
                        row['reference_'+key+'_mean']=metrics[key]['reference']['mean']
                    row['total_energy_change_eV']=metrics['energy_change_eV']['SMDP']
                    for band,spectrum in metrics['VDOS'].items():
                        for z,value in spectrum.get('elements',{}).items():row[f'{band}_Z{z}_overlap']=value['overlap']
                else:
                    metrics=json.loads((analysis/'motion.json').read_text())['metrics']
                    row['temperature_K_mean']=metrics['temperature_K']['SMDP']['mean']
                    row['reference_temperature_K_mean']=metrics['temperature_K']['reference']['mean']
                    for key,value in metrics['kinetic_partition'].items():
                        row[key+'_mean']=value['SMDP']['mean'];row['reference_'+key+'_mean']=value['reference']['mean']
                    for task in ['diffusion','vdos','full_energy']:
                        path=analysis/f'{task}.json'
                        if not path.exists():continue
                        record=json.loads(path.read_text())
                        if record.get('completed'):
                            row[task+'_metrics_json']=json.dumps(record['metrics'])
                rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT/'existing_production_results.csv',index=False)


if __name__=='__main__':main()
