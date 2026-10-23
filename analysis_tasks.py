"""Independent, resumable physical analyses. All trajectories are opened read-only."""
import argparse, json, os, time, traceback
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_quartz import stats, paired, spectrum_blocks, rdf_sample
from physics.units import KINETIC_TO_EV, KB_EV_PER_K


def save_json(path, obj):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False));temporary.replace(path)


class Context:
    def __init__(self,c,r,out):
        self.c,self.r,self.out=c,r,out
        self.completed=int(c.attrs['completed_steps']);thermal=int(c.attrs.get('thermalization_steps',0))
        self.production=self.completed>thermal;self.start=thermal if self.production else 0
        self.common=min(self.completed-self.start+1,(len(r['time_fs'])-1)//16+1)
        if self.common<2: raise ValueError('No complete common propagation step')
        self.z=c['atomic_numbers'][:];self.m=c['masses_u'][:];self.cell=c['cell'][:]
        np.testing.assert_array_equal(self.z,r['atomic_numbers'][:]);np.testing.assert_allclose(self.m,r['masses_u'][:])
        self.dof=3*len(self.m)-3
        self.idx=np.unique(np.linspace(0,self.common-1,min(2000,self.common),dtype=int))
        self.times=self.idx*16.;self.ci=self.start+self.idx
        self.cv=c['velocities'][self.ci].astype(float);self.rv=r['velocities_A_per_fs'][self.idx*16].astype(float)
        self.cke=self.ke(self.cv);self.rke=self.ke(self.rv)
    def ke(self,v): return .5*KINETIC_TO_EV*(self.m[None,:,None]*v*v).sum((1,2))


def thermo(x):
    result=dict(temperature_K=paired(2*x.cke/(x.dof*KB_EV_PER_K),2*x.rke/(x.dof*KB_EV_PER_K)),
        kinetic_energy_eV=paired(x.cke,x.rke),element_kinetic_energy_eV={},
        segment_join_derivative_RMS_A_fs=stats(x.c['segment_join_derivative_RMS_A_fs'][x.start:x.completed+1]),
        minimum_distance_A=float(x.c['minimum_distance_A'][:].min()),thermodynamic_sampling='2000 uniform production frames, matching R1')
    for z in np.unique(x.z):
        m=x.z==z
        result['element_kinetic_energy_eV'][str(z)]=paired(
            (.5*KINETIC_TO_EV*x.m[None,m,None]*x.cv[:,m]**2).sum((1,2)),
            (.5*KINETIC_TO_EV*x.m[None,m,None]*x.rv[:,m]**2).sum((1,2)))
    pd.DataFrame(dict(time_fs=x.times,SMDP_KE_eV=x.cke,reference_KE_eV=x.rke,
        SMDP_temperature_K=2*x.cke/(x.dof*KB_EV_PER_K),reference_temperature_K=2*x.rke/(x.dof*KB_EV_PER_K))).to_csv(x.out/'kinetic.csv',index=False)
    if x.c.attrs['ensemble']=='nve':
        q=x.c['positions'][x.ci];qr=x.r['positions_unwrapped_A'][x.idx*16]
        pd.DataFrame(dict(time_fs=x.times,position_RMSE_A=np.sqrt(((q-qr)**2).mean((1,2))),
            velocity_RMSE_A_fs=np.sqrt(((x.cv-x.rv)**2).mean((1,2))))).to_csv(x.out/'nve_trajectory_error.csv',index=False)
    return result


def energy_task(x):
    # Lazy import/initialization: missing MPI cannot prevent other tasks.
    from physics.bks import make_calculator, energy
    calc=make_calculator();cpe=[];rpe=[]
    for frame,rf in zip(x.ci,x.idx*16):
        cpe.append(energy(Atoms(numbers=x.z,positions=x.c['positions'][int(frame)],cell=x.cell,pbc=True),calc)[1])
        rpe.append(energy(Atoms(numbers=x.z,positions=x.r['positions_unwrapped_A'][int(rf)],cell=x.cell,pbc=True),calc)[1])
    calc.lmp.close();cpe=np.array(cpe);rpe=np.array(rpe)
    total=cpe+x.cke;rtotal=rpe+x.rke
    result=dict(potential_energy_eV=paired(cpe,rpe),energy_evaluation='Unchanged raw BKS buck/coul-dsf energy; offline only',
        total_energy_drift_meV_atom_ps=dict(SMDP=float(np.polyfit(x.times/1000,total,1)[0]*1000/len(x.m)),
            reference=float(np.polyfit(x.times/1000,rtotal,1)[0]*1000/len(x.m))),
        energy_change_eV=dict(SMDP=float(total[-1]-total[0]),reference=float(rtotal[-1]-rtotal[0])))
    data=dict(time_fs=x.times,SMDP_PE_eV=cpe,reference_PE_eV=rpe,SMDP_KE_eV=x.cke,reference_KE_eV=x.rke,
        SMDP_total_eV=total,reference_total_eV=rtotal,SMDP_temperature_K=2*x.cke/(x.dof*KB_EV_PER_K),
        reference_temperature_K=2*x.rke/(x.dof*KB_EV_PER_K))
    pd.DataFrame(data).to_csv(x.out/'thermodynamics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(x.times/1000,total-total[0],label='SMDP');axes[0].plot(x.times/1000,rtotal-rtotal[0],label='BKS')
    axes[0].set(xlabel='ps',ylabel='Total energy change (eV)');axes[0].legend()
    axes[1].hist(cpe,bins=40,density=True,alpha=.5,label='SMDP');axes[1].hist(rpe,bins=40,density=True,alpha=.5,label='BKS');axes[1].legend()
    fig.tight_layout();fig.savefig(x.out/'energy.png',dpi=160);plt.close(fig)
    if x.c.attrs['ensemble']=='nve':
        result['nve_windows']=energy_windows(data)
        fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
        for ax,key in zip(axes,['KE','PE','total']):
            ax.plot(x.times/1000,np.array(data['SMDP_'+key+'_eV'])-data['SMDP_'+key+'_eV'][0],label='SMDP')
            ax.plot(x.times/1000,np.array(data['reference_'+key+'_eV'])-data['reference_'+key+'_eV'][0],label='BKS')
            ax.set_ylabel('delta '+key+' (eV)');ax.legend()
        axes[-1].set_xlabel('ps');fig.tight_layout();fig.savefig(x.out/'energy_partition.png',dpi=160);plt.close(fig)
    return result


def energy_windows(data):
    t=np.asarray(data['time_fs'])/1000;rows=[]
    for a,b in [(0,1),(1,5),(5,20),(20,100)]:
        mask=(t>=a)&(t<b if b<100 else t<=b)
        if mask.sum()<2: continue
        row=dict(start_ps=a,end_ps=b,samples=int(mask.sum()),endpoint_convention='interpolated between identical sampled energy frames')
        for name in ['SMDP','reference']:
            row[name]={}
            for key in ['KE','PE','total']:
                y=np.asarray(data[name+'_'+key+'_eV'])
                row[name][key]=dict(mean_eV=float(y[mask].mean()),change_eV=float(np.interp(b,t,y)-np.interp(a,t,y)),
                    mean_change_from_initial_eV=float((y[mask]-y[0]).mean()))
            row[name]['temperature_K']=float(np.asarray(data[name+'_temperature_K'])[mask].mean())
        rows.append(row)
    return rows


def spectral(x,label):
    if label=='macro_16fs': cd,cs,cf,rs,dt=x.c['velocities'],x.start,x.common,16,16.
    else: cd,cs,cf,rs,dt=x.c['dense/velocities'],x.start*16,(x.common-1)*16+1,1,1.
    workers=getattr(x,'spectral_workers',1)
    cand=spectrum_blocks(cd,x.z,cs,cf,1,dt,cache=x.out/'spectral_cache'/f'{label}_candidate',workers=workers)
    ref=spectrum_blocks(x.r['velocities_A_per_fs'],x.z,0,cf,rs,dt,cache=x.out/'spectral_cache'/f'{label}_reference',workers=workers)
    if cand is None or ref is None: return {'VDOS':{label:dict(available=False,reason='less than 4.096 ps')}}
    freq,cp=cand;_,rp=ref;result=dict(nyquist_THz=float(freq[-1]),elements={})
    fig,axes=plt.subplots(2,1,figsize=(7,6));rows=[]
    for ax,z in zip(axes,sorted(cp)):
        a,b=cp[z].mean(0),rp[z].mean(0)
        result['elements'][str(z)]=dict(overlap=float(np.minimum(a,b).sum()),block_overlap=stats(np.minimum(cp[z],rp[z]).sum(1)),
            reference_split_half_overlap=float(np.minimum(rp[z][::2].mean(0),rp[z][1::2].mean(0)).sum()) if len(rp[z])>1 else None)
        ax.plot(freq,b,label='BKS');ax.plot(freq,a,label='SMDP');ax.set_title(f'Z={z}, {label}')
        ax.set_xlim(0,min(float(freq[-1]),80));ax.legend();ax.set_xlabel('THz')
        rows.extend(dict(element=z,frequency_THz=float(f),reference=float(v),SMDP=float(w)) for f,v,w in zip(freq,b,a))
    fig.tight_layout();fig.savefig(x.out/f'vdos_{label}.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(x.out/f'vdos_{label}.csv',index=False)
    return {'VDOS':{label:result}}


def rdf_task(x):
    ri=np.unique(np.linspace(0,x.common-1,min(512,x.common),dtype=int))
    cq=x.c['positions'][x.start+ri];rq=x.r['positions_unwrapped_A'][ri*16]
    rmax=min(6.,.49/np.linalg.norm(np.linalg.inv(x.cell),axis=0).max());result={};rows=[]
    fig,axes=plt.subplots(2,1,figsize=(7,6))
    for ax,(a,b) in zip(axes,[(14,8),(8,8)]):
        radius,cr=rdf_sample(cq,x.cell,x.z,a,b,rmax);_,rr=rdf_sample(rq,x.cell,x.z,a,b,rmax)
        cm,rm=cr.mean(0),rr.mean(0)
        mask=(radius>1)&(radius<2.2) if a==14 else (radius>2)&(radius<3.2)
        choices=np.flatnonzero(mask);ic=choices[np.argmax(cm[mask])];ir=choices[np.argmax(rm[mask])]
        result[f'{a}-{b}']=dict(RMSE=float(np.sqrt(np.mean((cm-rm)**2))),
            peak_position_A=dict(SMDP=float(radius[ic]),reference=float(radius[ir]),difference=float(radius[ic]-radius[ir])),
            peak_height=dict(SMDP=float(cm[ic]),reference=float(rm[ir]),difference=float(cm[ic]-rm[ir])),
            block_curve_RMSE=stats([np.sqrt(np.mean((u.mean(0)-v.mean(0))**2)) for u,v in zip(np.array_split(cr,min(8,len(cr))),np.array_split(rr,min(8,len(rr))))]))
        ax.plot(radius,rm,label='BKS');ax.plot(radius,cm,label='SMDP');ax.legend();ax.set_title(f'{a}-{b}');ax.set_xlabel('Angstrom')
        rows.extend(dict(pair=f'{a}-{b}',r_A=float(r),reference=float(v),SMDP=float(w)) for r,v,w in zip(radius,rm,cm))
    fig.tight_layout();fig.savefig(x.out/'rdf.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(x.out/'rdf.csv',index=False)
    return {'RDF':result}


def thermostat(x):
    if x.c.attrs['ensemble']!='nvt': return {'thermostat':{'applicable':False}}
    start=x.start+1;stop=x.completed+1
    post=x.c['temperature_K'][x.start:stop].astype(float)
    pre=x.c['pre_thermostat_temperature_K'][start:stop].astype(float)
    alpha=x.c['thermostat_alpha'][start:stop].astype(float)
    metadata=json.loads(x.c.attrs['thermostat_metadata']);decay=np.exp(-16/metadata['thermostat_tau_fs'])
    expected=decay*pre+(1-decay)*300;factor=x.dof*KB_EV_PER_K/2
    network=(pre-post[:-1])*factor;exchange=(post[1:]-pre)*factor
    result=dict(target_K=300,tau_fs=metadata['thermostat_tau_fs'],dt_fs=16,production_steps=len(pre),
        post_temperature_K=stats(post[1:]),pre_temperature_K=stats(pre),alpha=stats(alpha),
        network_delta_KE_eV=stats(network),csvr_delta_KE_eV=stats(exchange),
        post_step_delta_KE_eV=stats(np.diff(post)*factor),
        scaling_max_error_K=float(abs(post[1:]-pre*alpha**2).max()),
        conditional_mean_innovation_K=stats(post[1:]-expected),
        predicted_stationary_post_temperature_K=float(300+decay/(1-decay)*network.mean()/factor))
    if 'post_thermostat_species_KE_eV' in x.c:
        before=x.c['pre_thermostat_species_KE_eV'][start:stop].astype(float)
        after=x.c['post_thermostat_species_KE_eV'][x.start:stop].astype(float)
        result['species']={str(int(z)):dict(network_delta_KE_eV=stats(before[:,i]-after[:-1,i]),
            csvr_delta_KE_eV=stats(after[1:,i]-before[:,i]),pre_KE_eV=stats(before[:,i]),post_KE_eV=stats(after[1:,i]))
            for i,z in enumerate(x.c['kinetic_species'][:])}
    # Bounded time-block output, retaining per-step source arrays in original HDF5.
    rows=[]
    for ids in np.array_split(np.arange(len(pre)),100):
        rows.append(dict(time_ps=float(ids.mean()*.016),pre_temperature_K=float(pre[ids].mean()),post_temperature_K=float(post[1:][ids].mean()),
            network_delta_KE_eV=float(network[ids].mean()),csvr_delta_KE_eV=float(exchange[ids].mean())))
    pd.DataFrame(rows).to_csv(x.out/'thermostat_blocks.csv',index=False)
    return {'thermostat':result}


TASKS={'thermo':thermo,'rdf':rdf_task,'energy':energy_task,'thermostat':thermostat,
       'vdos_macro':lambda x:spectral(x,'macro_16fs'),'vdos_dense':lambda x:spectral(x,'queried_1fs')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    p.add_argument('--spectral-workers',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.spectral_workers<1:raise ValueError('spectral-workers must be positive')
    identity={k:dict(path=str(v.resolve()),size=v.stat().st_size,mtime_ns=v.stat().st_mtime_ns) for k,v in [('candidate',a.candidate),('reference',a.reference)]}
    manifest=a.output/'analysis_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity: raise ValueError('Sources changed; use a new output directory')
    save_json(manifest,identity)
    with h5py.File(a.candidate,'r') as c,h5py.File(a.reference,'r') as r:
        x=Context(c,r,a.output)
        x.spectral_workers=a.spectral_workers
        base=dict(candidate=str(a.candidate),reference=str(a.reference),production_available=x.production,
            trajectory_completed=x.completed>=int(c.attrs.get('thermalization_steps',0))+int(c.attrs['requested_steps']),
            stable=bool(c.attrs['stable']),failure=str(c.attrs['failure_reason']),evaluation_phase='production' if x.production else 'failed_thermalization_diagnostic',
            duration_ps=(x.common-1)*.016,requested_production_ps=int(c.attrs['requested_steps'])*.016)
        def consolidate():
            result=dict(base);status={}
            for name in TASKS:
                path=a.output/(name+'.json')
                if not path.exists():status[name]={'completed':False,'failure_reason':'not run'};continue
                record=json.loads(path.read_text());status[name]={k:record[k] for k in ['completed','failure_reason','elapsed_s']}
                if record['completed']:
                    payload=record['metrics']
                    for k,v in payload.items():
                        if k=='VDOS':result.setdefault(k,{}).update(v)
                        else:result[k]=v
            result['task_status']=status;result['physical_analysis_completed']=all(v['completed'] for v in status.values())
            result['analysis_failure_reason']={k:v['failure_reason'] for k,v in status.items() if not v['completed']}
            save_json(a.output/'metrics.json',result);return result
        for name in a.tasks:
            target=a.output/(name+'.json')
            if target.exists() and json.loads(target.read_text()).get('completed'):continue
            started=time.perf_counter()
            try:
                payload=TASKS[name](x);record=dict(completed=True,failure_reason=None,metrics=payload)
            except Exception as error:
                text=traceback.format_exc();(a.output/(name+'.traceback.txt')).write_text(text)
                print(text,flush=True);record=dict(completed=False,failure_reason=f'{type(error).__name__}: {error}',metrics={})
            record['elapsed_s']=time.perf_counter()-started;save_json(target,record);consolidate()
            print(json.dumps(dict(task=name,completed=record['completed'],elapsed_s=record['elapsed_s'])),flush=True)
        result=consolidate()
        print(json.dumps({k:result[k] for k in ['trajectory_completed','physical_analysis_completed','analysis_failure_reason']}),flush=True)
        if any(not result['task_status'][name]['completed'] for name in a.tasks): raise SystemExit(1)


if __name__=='__main__': main()
