"""Independent, resumable SPC/Fw trajectory analyses; all inputs read-only."""
import argparse
import json
import time
import traceback
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physics.statistics import paired, rdf_sample, spectrum_blocks, stats
from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from physics.water_motion import motion_partition
from physics.water_motion import molecular_geometry
from physics.diffusion import time_averaged_msd


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class Context:
    def __init__(self, candidate, reference, out, frames):
        self.c, self.r, self.out = candidate, reference, out
        self.model_label = 'DSRT'
        self.dt = float(candidate.attrs['timestep_fs'])
        self.stop = int(candidate.attrs['completed_steps'])
        self.start = int(candidate.attrs.get('thermalization_steps', 0))
        if self.stop <= self.start:
            raise ValueError('No complete production trajectory')
        reference_dt = float(reference.attrs['output_interval_fs'])
        self.stride = round(self.dt/reference_dt)
        if not np.isclose(self.stride*reference_dt, self.dt):
            raise ValueError('Nonintegral common sampling')
        self.common = min(self.stop-self.start+1, (len(reference['time_fs'])-1)//self.stride+1)
        self.z, self.m = candidate['atomic_numbers'][:], candidate['masses_u'][:]
        self.cell = candidate['cell'][:]
        self.mol = reference['molecule_ids'][:]
        np.testing.assert_array_equal(self.z, reference['atomic_numbers'][:])
        np.testing.assert_allclose(self.m, reference['masses_u'][:])
        self.idx = np.unique(np.linspace(0, self.common-1, min(frames, self.common), dtype=int))
        self.ci, self.ri = self.start+self.idx, self.idx*self.stride
        self.time = self.idx*self.dt/1000
        self.q = candidate['positions'][self.ci].astype(float)
        self.v = candidate['velocities'][self.ci].astype(float)
        self.rq = reference['positions_unwrapped_A'][self.ri].astype(float)
        self.rv = reference['velocities_A_per_fs'][self.ri].astype(float)
        self.motion = motion_partition(self.q, self.v, self.cell, self.z, self.m, self.mol)
        self.rmotion = motion_partition(self.rq, self.rv, self.cell, self.z, self.m, self.mol)


def motion_task(x):
    result = {'kinetic_partition': {}, 'species_kinetic_energy_eV': {}}
    rows = {'time_ps': x.time}
    for key in ['translation_eV', 'rotation_eV', 'internal_eV', 'total_KE_eV']:
        result['kinetic_partition'][key] = paired(x.motion[key], x.rmotion[key])
        rows['model_'+key], rows['reference_'+key] = x.motion[key], x.rmotion[key]
    dof = 3*len(x.m)-3
    result['temperature_K'] = paired(2*x.motion['total_KE_eV']/(dof*KB_EV_PER_K),
                                     2*x.rmotion['total_KE_eV']/(dof*KB_EV_PER_K))
    for z in sorted(set(x.z)):
        mask = x.z == z
        ke = lambda velocity: .5*KINETIC_TO_EV*(x.m[None, mask, None]*velocity[:, mask]**2).sum((1, 2))
        result['species_kinetic_energy_eV'][str(z)] = paired(ke(x.v), ke(x.rv))
    for key in ['bond_lengths_A', 'angles_deg']:
        result[key] = paired(x.motion[key].reshape(-1), x.rmotion[key].reshape(-1))
    result['sampling'] = dict(frames=len(x.idx), common_duration_ps=float(x.time[-1]),
                               frame_statistics_correlated=True)
    pd.DataFrame(rows).to_csv(x.out/'motion.csv', index=False)
    result['windows'] = []
    for a, b in [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100)]:
        mask = (x.time >= a) & (x.time <= b)
        if mask.sum() < 2:
            continue
        result['windows'].append(dict(start_ps=a, stop_ps=min(b, float(x.time[-1])),
            **{key: paired(x.motion[key][mask], x.rmotion[key][mask])
               for key in ['translation_eV', 'rotation_eV', 'internal_eV', 'total_KE_eV']}))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.flat, ['translation_eV', 'rotation_eV', 'internal_eV', 'total_KE_eV']):
        ax.plot(x.time, x.motion[key], label=x.model_label); ax.plot(x.time, x.rmotion[key], label='Reference', alpha=.65)
        ax.set(xlabel='ps', ylabel=key); ax.legend()
    fig.tight_layout(); fig.savefig(x.out/'motion.png', dpi=160); plt.close(fig)
    return result


def energy_task(x, topology):
    from physics.spcfw import WaterReference, verify_reference
    calc = WaterReference(topology, x.cell, x.z)
    try:
        frames = np.unique(np.linspace(0, len(x.r['time_fs'])-1, 8, dtype=int))
        verification = verify_reference(x.r, calc, frames)
        accepted = (max(abs(r['energy_error_eV']) for r in verification) < 1e-3 and
                    max(r.get('force_max_error_eV_A', 0.) for r in verification) < .02)
        save(x.out/'reference_verification.json', dict(accepted=accepted, frames=verification,
            tolerances=dict(energy_eV=1e-3, max_force_eV_A=.02),
            rationale='Allows rounded raw dump coordinates/forces; no energy recentering'))
        if not accepted:
            raise ValueError('Reference energy/force verification failed; see reference_verification.json')
        rows = []
        for i, (q, rq) in enumerate(zip(x.q, x.rq)):
            ce, re = calc.evaluate(q), calc.evaluate(rq)
            row = dict(time_ps=float(x.time[i]))
            for key in ce:
                row['model_'+key], row['reference_'+key] = ce[key], re[key]
            row['model_ke'], row['reference_ke'] = x.motion['total_KE_eV'][i], x.rmotion['total_KE_eV'][i]
            row['model_total'] = ce['pe']+row['model_ke']
            row['reference_total'] = re['pe']+row['reference_ke']
            rows.append(row)
            if i % 250 == 0:
                print(json.dumps(dict(task='energy', frame=i, total=len(x.q))), flush=True)
    finally:
        calc.close()
    df = pd.DataFrame(rows); df.to_csv(x.out/'energy.csv', index=False)
    result = dict(reference_verified=True, common_duration_ps=float(x.time[-1]), components={})
    for key in ['pe', 'ebond', 'eangle', 'evdwl', 'ecoul', 'elong', 'nonbonded', 'ke', 'total']:
        a, b = df['model_'+key].to_numpy(), df['reference_'+key].to_numpy()
        result['components'][key] = dict(distribution=paired(a, b),
            change_eV=dict(model=float(a[-1]-a[0]), reference=float(b[-1]-b[0])),
            slope_eV_ps=dict(model=float(np.polyfit(x.time, a, 1)[0]),
                             reference=float(np.polyfit(x.time, b, 1)[0])))
    result['windows'] = []
    for a, b in [(0, 1), (1, 5), (5, 20), (20, 50)]:
        if a >= x.time[-1]:
            continue
        end = min(b, float(x.time[-1])); window = dict(start_ps=a, stop_ps=end)
        for key in ['ke', 'ebond', 'eangle', 'nonbonded', 'pe', 'total']:
            window[key] = {name: float(np.interp(end, x.time, df[name+'_'+key])-
                                      np.interp(a, x.time, df[name+'_'+key]))
                           for name in ['model', 'reference']}
        result['windows'].append(window)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, keys in zip(axes, [['ke', 'pe', 'total'], ['ebond', 'eangle', 'nonbonded'], ['total']]):
        for key in keys:
            for name, style in [('model', '-'), ('reference', '--')]:
                y = df[name+'_'+key].to_numpy()
                ax.plot(x.time, y-y[0], style, label=name+' '+key, alpha=.8)
        ax.set(ylabel='Change (eV)'); ax.legend()
    axes[-1].set_xlabel('ps'); fig.tight_layout(); fig.savefig(x.out/'energy.png', dpi=160); plt.close(fig)
    return result


def rdf_task(x):
    selected = np.unique(np.linspace(0, len(x.q)-1, min(512, len(x.q)), dtype=int))
    rmax = min(6., .49/np.linalg.norm(np.linalg.inv(x.cell), axis=0).max())
    result, rows = {}, []
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    for ax, (a, b, lo, hi) in zip(axes, [(8, 8, 2.4, 3.3), (8, 1, .8, 1.3), (1, 1, 1.2, 2.)]):
        radius, c = rdf_sample(x.q[selected], x.cell, x.z, a, b, rmax)
        _, r = rdf_sample(x.rq[selected], x.cell, x.z, a, b, rmax)
        cm, rm = c.mean(0), r.mean(0); peak = np.flatnonzero((radius > lo) & (radius < hi))
        ic, ir = peak[np.argmax(cm[peak])], peak[np.argmax(rm[peak])]
        result[f'{a}-{b}'] = dict(RMSE=float(np.sqrt(np.mean((cm-rm)**2))),
            peak_search_A=[lo, hi], includes_intramolecular_pairs=True,
            peak_position_A=dict(model=float(radius[ic]), reference=float(radius[ir])),
            peak_height=dict(model=float(cm[ic]), reference=float(rm[ir])))
        ax.plot(radius, cm, label=x.model_label); ax.plot(radius, rm, label='Reference'); ax.legend()
        ax.set(xlabel='Å', ylabel='g(r)', title=f'{a}-{b}')
        rows.extend(dict(pair=f'{a}-{b}',r_A=float(v),model=float(u),reference=float(w))
                    for v,u,w in zip(radius,cm,rm))
    pd.DataFrame(rows).to_csv(x.out/'rdf.csv', index=False)
    fig.tight_layout(); fig.savefig(x.out/'rdf.png',dpi=160); plt.close(fig)
    return result


def vdos_task(x):
    c = spectrum_blocks(x.c['velocities'], x.z, x.start, x.common, 1, x.dt,
                        cache=x.out/'spectral_cache/model')
    r = spectrum_blocks(x.r['velocities_A_per_fs'], x.z, 0, x.common, x.stride, x.dt,
                        cache=x.out/'spectral_cache/reference')
    if c is None or r is None:
        return dict(available=False, reason='Less than 4.096 ps')
    frequency, cp = c; _, rp = r
    result = dict(nyquist_THz=float(frequency[-1]), cadence_fs=x.dt, elements={},
                  interpretation=('Matched sampled NVT production spectra' if x.c.attrs['ensemble']=='nvt'
                                  else 'Matched sampled spectra; NVE heating is nonstationary'))
    fig, axes = plt.subplots(2, 1, figsize=(8, 7)); rows = []
    for ax, z in zip(axes, sorted(cp)):
        cm, rm = cp[z].mean(0), rp[z].mean(0)
        result['elements'][str(z)] = dict(overlap=float(np.minimum(cm,rm).sum()),
            block_overlap=stats(np.minimum(cp[z],rp[z]).sum(axis=1)))
        ax.plot(frequency,cm,label=x.model_label); ax.plot(frequency,rm,label='Reference')
        ax.set(xlabel='THz', title=f'Z={z}'); ax.legend()
        rows.extend(dict(element=int(z), frequency_THz=float(f), model=float(a), reference=float(b))
                    for f,a,b in zip(frequency,cm,rm))
    pd.DataFrame(rows).to_csv(x.out/'vdos.csv',index=False)
    fig.tight_layout(); fig.savefig(x.out/'vdos.png',dpi=160); plt.close(fig)
    # Explicit normalized VACF on eight 16.384 ps windows spread over the
    # production interval. Use the same uncentered atom-averaged dot product
    # as reference_vdos, retaining lags through 1 ps.
    block_frames=min(int(16384/x.dt),x.common)
    offsets=np.unique(np.linspace(0,x.common-block_frames,min(8,x.common//block_frames),dtype=int))
    lags=min(int(1000/x.dt),block_frames-1)
    vacf={name:{int(z):[] for z in np.unique(x.z)} for name in ['model','reference']}
    for offset in offsets:
        for name,dataset,start,stride in [('model',x.c['velocities'],x.start,1),
                                          ('reference',x.r['velocities_A_per_fs'],0,x.stride)]:
            values=dataset[start+offset*stride:start+(offset+block_frames)*stride:stride].astype(float)
            for z in vacf[name]:
                a=values[:,x.z==z].reshape(block_frames,-1)
                f=np.fft.rfft(a,n=2*block_frames,axis=0)
                ac=np.fft.irfft(np.sum(f.conj()*f,axis=1),n=2*block_frames)[:lags+1]
                ac/=np.arange(block_frames,block_frames-lags-1,-1)
                vacf[name][z].append(ac/ac[0])
    fig,axes=plt.subplots(2,1,figsize=(8,7));rows=[];result['VACF']={}
    for ax,z in zip(axes,sorted(vacf['model'])):
        a=np.mean(vacf['model'][z],axis=0);b=np.mean(vacf['reference'][z],axis=0)
        t=np.arange(len(a))*x.dt
        result['VACF'][str(z)]=dict(normalized_curve_RMSE=float(np.sqrt(np.mean((a-b)**2))),
            windows=len(offsets),maximum_lag_fs=float(t[-1]))
        ax.plot(t,a,label=x.model_label);ax.plot(t,b,label='Reference');ax.legend()
        ax.set(xlabel='Lag (fs)',ylabel='Normalized VACF',title=f'Z={z}')
        rows.extend(dict(element=int(z),lag_fs=float(v),model=float(u),reference=float(w))
                    for v,u,w in zip(t,a,b))
    pd.DataFrame(rows).to_csv(x.out/'vacf.csv',index=False)
    fig.tight_layout();fig.savefig(x.out/'vacf.png',dpi=160);plt.close(fig)
    return result


def full_energy_task(x, topology):
    """Complete candidate history, including times beyond reference coverage."""
    from physics.spcfw import WaterReference, verify_reference
    idx = np.unique(np.linspace(x.start,x.stop,min(2000,x.stop-x.start+1),dtype=int))
    times = (idx-x.start)*x.dt/1000
    q = x.c['positions'][idx].astype(float); v = x.c['velocities'][idx].astype(float)
    motion = motion_partition(q,v,x.cell,x.z,x.m,x.mol)
    calculator = WaterReference(topology,x.cell,x.z)
    try:
        verification = verify_reference(x.r,calculator,[0,len(x.r['time_fs'])-1])
        if max(abs(r['energy_error_eV']) for r in verification)>=1e-3:
            raise ValueError('Full-energy reference verification failed')
        rows=[]
        for i,positions in enumerate(q):
            energy=calculator.evaluate(positions)
            rows.append(dict(time_ps=float(times[i]),**energy,
                **{k:float(motion[k][i]) for k in ['translation_eV','rotation_eV','internal_eV','total_KE_eV']},
                total=energy['pe']+float(motion['total_KE_eV'][i])))
            if i%500==0:print(json.dumps(dict(task='full_energy',frame=i,total=len(q))),flush=True)
    finally:calculator.close()
    data=pd.DataFrame(rows); data.to_csv(x.out/'full_candidate_energy.csv',index=False)
    result=dict(duration_ps=float(times[-1]),reference_coverage_ps=float(x.time[-1]),
        components={k:dict(change_eV=float(data[k].iloc[-1]-data[k].iloc[0]),
            statistics=stats(data[k]),slope_eV_ps=float(np.polyfit(times,data[k],1)[0]))
            for k in data if k!='time_ps'}, windows=[])
    for a,b in [(0,1),(1,5),(5,20),(20,50),(50,100)]:
        if a>=times[-1]:continue
        end=min(b,float(times[-1]))
        result['windows'].append(dict(start_ps=a,stop_ps=end,change_eV={
            k:float(np.interp(end,times,data[k])-np.interp(a,times,data[k]))
            for k in ['pe','total_KE_eV','total','ebond','eangle','nonbonded',
                      'translation_eV','rotation_eV','internal_eV']}))
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for ax,keys in zip(axes,[['pe','total_KE_eV','total'],['ebond','eangle','nonbonded']]):
        for key in keys:ax.plot(times,data[key]-data[key].iloc[0],label=key)
        ax.axvline(x.time[-1],color='grey',linestyle=':',label='Reference coverage ends')
        ax.set_ylabel('Change (eV)');ax.legend()
    axes[-1].set_xlabel('ps');fig.tight_layout();fig.savefig(x.out/'full_candidate_energy.png',dpi=160);plt.close(fig)
    return result


def thermostat_task(x):
    if x.c.attrs['ensemble']!='nvt':return dict(applicable=False)
    start,stop=x.start+1,x.stop+1
    post=x.c['temperature_K'][x.start:stop].astype(float)
    pre=x.c['pre_thermostat_temperature_K'][start:stop].astype(float)
    alpha=x.c['thermostat_alpha'][start:stop].astype(float)
    meta=json.loads(x.c.attrs['thermostat_metadata']);dof=3*len(x.m)-3
    scale=dof*KB_EV_PER_K/2;decay=np.exp(-x.dt/meta['thermostat_tau_fs'])
    result=dict(target_K=meta['temperature_K'],tau_fs=meta['thermostat_tau_fs'],dt_fs=x.dt,
        production_steps=len(pre),pre_temperature_K=stats(pre),post_temperature_K=stats(post[1:]),
        alpha=stats(alpha),network_delta_KE_eV=stats((pre-post[:-1])*scale),
        csvr_delta_KE_eV=stats((post[1:]-pre)*scale),
        scaling_max_error_K=float(abs(post[1:]-pre*alpha**2).max()),
        conditional_mean_innovation_K=stats(post[1:]-(decay*pre+(1-decay)*meta['temperature_K'])))
    if 'pre_thermostat_species_KE_eV' in x.c:
        a=x.c['pre_thermostat_species_KE_eV'][start:stop]
        b=x.c['post_thermostat_species_KE_eV'][x.start:stop]
        result['species']={str(int(z)):dict(network_delta_KE_eV=stats(a[:,i]-b[:-1,i]),
            csvr_delta_KE_eV=stats(b[1:,i]-a[:,i]),mean_kinetic_eV=float(b[1:,i].mean()))
            for i,z in enumerate(x.c['kinetic_species'][:])}
    return result


def diffusion_task(x):
    if x.c.attrs['ensemble']!='nvt':
        return dict(applicable=False,reason='Nonstationary NVE heating; no equilibrium diffusion claim')
    # Frozen before inspecting the NVT result: 50 fs sampling, eight time
    # blocks, molecular COM MSD fit over 10--100 ps in every block.
    sampling_fs=50.;stride=round(sampling_fs/x.dt)
    if not np.isclose(stride*x.dt,sampling_fs):raise ValueError('Invalid MSD cadence')
    frames=(x.common-1)//stride+1
    q=x.c['positions'][x.start:x.start+frames*stride:stride].astype(float)
    rq=x.r['positions_unwrapped_A'][:frames*stride*x.stride:stride*x.stride].astype(float)

    def molecular_centres(values):
        ids,bonds,_,_=molecular_geometry(values,x.cell,x.z,x.mol)
        local=x.m[ids[:,1:],None]*bonds
        com=values[:,ids[:,0]]+local.sum(axis=-2)/x.m[ids].sum(axis=-1)[:,None]
        return com-com.mean(axis=1,keepdims=True)

    a,b=molecular_centres(q),molecular_centres(rq)
    results={};tables=[];fig,ax=plt.subplots(figsize=(8,5))
    for label,values in [('model',a),('reference',b)]:
        blocks=np.array_split(values,8);fits=[];curves=[]
        for block in blocks:
            msd=time_averaged_msd(block);t=np.arange(len(msd))*sampling_fs/1000
            mask=(t>=10)&(t<=100)
            if t[-1]<100:raise ValueError('MSD block shorter than frozen 100 ps fit limit')
            slope,intercept=np.polyfit(t[mask],msd[mask],1)
            residual=msd[mask]-(slope*t[mask]+intercept)
            denominator=np.sum((msd[mask]-msd[mask].mean())**2)
            fits.append(dict(diffusion_m2_s=float(slope/6*1e-8),slope_A2_ps=float(slope),
                             intercept_A2=float(intercept),R2=float(1-np.sum(residual**2)/denominator)))
            curves.append(msd)
        length=min(map(len,curves));mean=np.mean([v[:length] for v in curves],axis=0)
        t=np.arange(length)*sampling_fs/1000
        results[label]=dict(diffusion_m2_s=stats([v['diffusion_m2_s'] for v in fits]),block_fits=fits)
        ax.plot(t,mean,label=label)
        tables.extend(dict(source=label,time_ps=float(u),mean_MSD_A2=float(v)) for u,v in zip(t,mean))
    ax.set(xlim=(0,150),xlabel='Lag (ps)',ylabel='Molecular COM MSD (Å²)');ax.legend()
    fig.tight_layout();fig.savefig(x.out/'msd.png',dpi=160);plt.close(fig)
    pd.DataFrame(tables).to_csv(x.out/'msd.csv',index=False)
    return dict(results=results,fit_window_ps=[10,100],sampling_fs=sampling_fs,blocks=8,
                definition='Time-origin averaged unwrapped molecular COM; global COM drift removed')


def main():
    p = argparse.ArgumentParser()
    for key in ['candidate','reference','topology','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tasks',nargs='+',default=['motion','energy','rdf','vdos','full_energy','thermostat','diffusion'],
                   choices=['motion','energy','rdf','vdos','full_energy','thermostat','diffusion','vdos_dense'])
    p.add_argument('--frames',type=int,default=2000)
    p.add_argument('--model-label',default='DSRT')
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    identity = {key: dict(path=str(value.resolve()), size=value.stat().st_size,
                         mtime_ns=value.stat().st_mtime_ns)
                for key,value in [('candidate',args.candidate),('reference',args.reference),('topology',args.topology)]}
    identity['frames'] = args.frames
    if args.model_label!='DSRT':
        identity['model_label'] = args.model_label
    manifest = args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Sources/settings changed: use a different analysis output')
    save(manifest, identity)
    with h5py.File(args.candidate,'r') as c, h5py.File(args.reference,'r') as r:
        x = Context(c,r,args.output,args.frames)
        x.model_label = args.model_label
        tasks = dict(motion=motion_task, energy=lambda x:energy_task(x,args.topology),
                     rdf=rdf_task, vdos=vdos_task,full_energy=lambda x:full_energy_task(x,args.topology),
                     thermostat=thermostat_task,diffusion=diffusion_task)
        if 'vdos_dense' in args.tasks:
            from analysis_tasks import spectral
            tasks['vdos_dense'] = lambda context: spectral(context, 'dense_1fs')
        for name in args.tasks:
            dest = args.output/(name+'.json')
            if dest.exists() and json.loads(dest.read_text()).get('completed'):
                continue
            tic = time.perf_counter()
            try:
                record = dict(completed=True, metrics=tasks[name](x), failure_reason=None)
            except Exception as e:
                (args.output/(name+'.traceback.txt')).write_text(traceback.format_exc())
                record = dict(completed=False, metrics={}, failure_reason=f'{type(e).__name__}: {e}')
            record['elapsed_s'] = time.perf_counter()-tic; save(dest, record)
            print(json.dumps(dict(task=name, **{k:record[k] for k in ['completed','failure_reason','elapsed_s']})),flush=True)
        status = {name:json.loads((args.output/(name+'.json')).read_text()) for name in tasks
                  if (args.output/(name+'.json')).exists()}
        save(args.output/'status.json', dict(trajectory_completed=int(c.attrs['completed_steps']) >=
            int(c.attrs['requested_steps'])+int(c.attrs.get('thermalization_steps',0)),
            physical_analysis_completed=all(name in status and status[name]['completed'] for name in args.tasks),
            analysis_failure_reason={k:v['failure_reason'] for k,v in status.items() if not v['completed']},
            common_duration_ps=float(x.time[-1]),requested_tasks=args.tasks,
            task_status={k:{key:v[key] for key in ['completed','failure_reason']} for k,v in status.items()}))
        if not all(v['completed'] for v in status.values()):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
