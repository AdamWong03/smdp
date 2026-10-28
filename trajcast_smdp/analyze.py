"""Same reference MD/BKS and existing metric primitives, at 30 fs and 1 fs.

Run in the existing MLIP+MPICH analysis environment, not the neural venv.
Tasks persist independently so a failed energy calculation cannot hide spectra.
"""
import argparse,json,traceback
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_quartz import paired,stats,spectrum_blocks,rdf_sample
from physics.units import KINETIC_TO_EV,KB_EV_PER_K

RUN=Path('/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028')


def main(run,tag):
    out=Path(run)/tag;meta=json.loads((out/'run.json').read_text());analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    if not (out/'trajectory.h5').exists():
        (analysis/'status.json').write_text(json.dumps(dict(trajectory_completed=False,
            physical_analysis_completed=False,analysis_failure_reason=meta.get('failure_reason','No trajectory')),indent=2))
        return
    with h5py.File(out/'trajectory.h5') as c,h5py.File(meta['reference_path']) as r:
        done=int(c.attrs['completed_steps']);warm=int(c.attrs['thermalization_steps'])
        production=done>warm;start=warm if production else 0
        common=min(done-start+1,(len(r['time_fs'])-1)//30+1)
        status=dict(trajectory_completed=meta['trajectory_completed'],production_available=production,
            physical_analysis_completed=False,completed_steps=done,actual_production_ps=max(0,done-warm)*.03,
            reference=str(r.filename),tasks={})
        if common<2:
            status['analysis_failure_reason']='Insufficient completed frames'
            (analysis/'status.json').write_text(json.dumps(status,indent=2));return
        z=c['atomic_numbers'][:];m=c['masses_u'][:];cell=c['cell'][:];dof=3*len(m)-3
        np.testing.assert_array_equal(z,r['atomic_numbers'][:]);np.testing.assert_allclose(m,r['masses_u'][:])
        idx=np.unique(np.linspace(0,common-1,min(2000,common),dtype=int));ci=idx+start
        cv=c['velocities'][ci].astype(float);rv=r['velocities_A_per_fs'][idx*30].astype(float)
        ke=lambda v:.5*KINETIC_TO_EV*(m[None,:,None]*v*v).sum((1,2))
        ck,rk=ke(cv),ke(rv);time_fs=idx*30.
        def thermo():
            result=dict(temperature_K=paired(2*ck/(dof*KB_EV_PER_K),2*rk/(dof*KB_EV_PER_K)),
                kinetic_energy_eV=paired(ck,rk),minimum_distance_A=float(c['minimum_distance_A'][1:done+1].min()),
                per_element={})
            for element in np.unique(z):
                mask=z==element
                a=.5*KINETIC_TO_EV*(m[None,mask,None]*cv[:,mask]**2).sum((1,2))
                b=.5*KINETIC_TO_EV*(m[None,mask,None]*rv[:,mask]**2).sum((1,2))
                result['per_element'][str(element)]=paired(a,b)
            if meta['ensemble']=='nvt':
                pre=c['pre_thermostat_temperature_K'][start+1:done+1]
                post=c['temperature_K'][start+1:done+1];previous=c['temperature_K'][start:done]
                factor=dof*KB_EV_PER_K/2
                result['thermostat']=dict(pre_temperature=stats(pre),post_temperature=stats(post),
                    alpha=stats(c['thermostat_alpha'][start+1:done+1]),
                    network_delta_KE_eV=stats((pre-previous)*factor),csvr_delta_KE_eV=stats((post-pre)*factor))
            else:
                cq=c['positions'][ci];rq=r['positions_unwrapped_A'][idx*30]
                pd.DataFrame(dict(time_fs=time_fs,position_RMSE_A=np.sqrt(((cq-rq)**2).mean((1,2))),
                    velocity_RMSE_A_fs=np.sqrt(((cv-rv)**2).mean((1,2))))).to_csv(analysis/'nve_errors.csv',index=False)
            return result
        def energies():
            from physics.bks import make_calculator,energy
            calc=make_calculator();pe=[];ref=[]
            try:
                for a,b in zip(ci,idx*30):
                    pe.append(energy(Atoms(numbers=z,positions=c['positions'][int(a)],cell=cell,pbc=True),calc)[1])
                    ref.append(energy(Atoms(numbers=z,positions=r['positions_unwrapped_A'][int(b)],cell=cell,pbc=True),calc)[1])
            finally:
                if calc.lmp is not None: calc.lmp.close()
            pe=np.asarray(pe);ref=np.asarray(ref);total=pe+ck;rt=ref+rk
            pd.DataFrame(dict(time_fs=time_fs,SMDP_PE_eV=pe,reference_PE_eV=ref,SMDP_total_eV=total,
                reference_total_eV=rt,SMDP_KE_eV=ck,reference_KE_eV=rk)).to_csv(analysis/'energy.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            axes[0].plot(time_fs/1000,total-total[0],label='SMDP');axes[0].plot(time_fs/1000,rt-rt[0],label='BKS')
            axes[0].set(xlabel='ps',ylabel='Total energy change / eV');axes[0].legend()
            axes[1].hist(pe,bins=40,density=True,alpha=.5,label='SMDP');axes[1].hist(ref,bins=40,density=True,alpha=.5,label='BKS');axes[1].legend()
            fig.tight_layout();fig.savefig(analysis/'energy.png',dpi=160);plt.close(fig)
            return dict(potential_energy_eV=paired(pe,ref),energy_convention='raw BKS DSF, same parameters/units as own MD',
                energy_change_eV=dict(SMDP=float(total[-1]-total[0]),reference=float(rt[-1]-rt[0])),
                drift_meV_atom_ps=dict(SMDP=float(np.polyfit(time_fs/1000,total,1)[0]*1000/len(m)),
                    reference=float(np.polyfit(time_fs/1000,rt,1)[0]*1000/len(m))))
        def spectra(dense):
            dt=1 if dense else 30;label='dense_1fs' if dense else 'macro_30fs'
            cd=c['dense/velocities'] if dense else c['velocities'];begin=start*30 if dense else start
            length=(common-1)*30+1 if dense else common;stride=1 if dense else 30
            a=spectrum_blocks(cd,z,begin,length,1,dt,cache=analysis/'spectral_cache'/label/'candidate',workers=4)
            b=spectrum_blocks(r['velocities_A_per_fs'],z,0,length,stride,dt,cache=analysis/'spectral_cache'/label/'reference',workers=4)
            if a is None or b is None: return dict(available=False,reason='less than 4.096 ps')
            freq,cp=a;_,rp=b;result=dict(dt_fs=dt,nyquist_THz=500/dt,elements={});rows=[]
            fig,axes=plt.subplots(2,1,figsize=(7,6))
            for ax,element in zip(axes,sorted(cp)):
                p,q=cp[element].mean(0),rp[element].mean(0)
                result['elements'][str(element)]=dict(overlap=float(np.minimum(p,q).sum()),
                    block_overlap=stats(np.minimum(cp[element],rp[element]).sum(1)))
                ax.plot(freq,p,label='SMDP');ax.plot(freq,q,label='BKS');ax.set_xlim(0,min(80,500/dt));ax.legend()
                ax.set_title(f'{label}: Z={element}');ax.set_xlabel('THz')
                rows.extend(dict(element=element,frequency_THz=float(f),SMDP=float(x),reference=float(y)) for f,x,y in zip(freq,p,q))
            fig.tight_layout();fig.savefig(analysis/f'{label}.png',dpi=160);plt.close(fig)
            pd.DataFrame(rows).to_csv(analysis/f'{label}.csv',index=False);return result
        def rdf():
            ii=np.unique(np.linspace(0,common-1,min(512,common),dtype=int));rows=[];result={}
            rmax=min(6.,.49/np.linalg.norm(np.linalg.inv(cell),axis=0).max())
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,(a,b) in zip(axes,[(14,8),(8,8)]):
                radius,cp=rdf_sample(c['positions'][start+ii],cell,z,a,b,rmax)
                _,rp=rdf_sample(r['positions_unwrapped_A'][ii*30],cell,z,a,b,rmax)
                p,q=cp.mean(0),rp.mean(0);mask=(radius>1)&(radius<2.2) if a==14 else (radius>2)&(radius<3.2)
                ix=np.flatnonzero(mask);ip=ix[np.argmax(p[mask])];ir=ix[np.argmax(q[mask])]
                result[f'{a}-{b}']=dict(RMSE=float(np.mean((p-q)**2)**.5),
                    peak_position_A=dict(SMDP=float(radius[ip]),reference=float(radius[ir])),
                    peak_height=dict(SMDP=float(p[ip]),reference=float(q[ir])))
                ax.plot(radius,p,label='SMDP');ax.plot(radius,q,label='BKS');ax.legend();ax.set_title(f'{a}-{b}')
                rows.extend(dict(pair=f'{a}-{b}',radius_A=float(r),SMDP=float(x),reference=float(y)) for r,x,y in zip(radius,p,q))
            fig.tight_layout();fig.savefig(analysis/'rdf.png',dpi=160);plt.close(fig)
            pd.DataFrame(rows).to_csv(analysis/'rdf.csv',index=False);return result
        for name,task in [('thermo',thermo),('energy',energies),('macro_30fs',lambda:spectra(False)),('dense_1fs',lambda:spectra(True)),('rdf',rdf)]:
            target=analysis/f'{name}.json'
            try:
                if not target.exists(): target.write_text(json.dumps(task(),indent=2,allow_nan=False)+'\n')
                status['tasks'][name]=dict(completed=True)
            except Exception:
                status['tasks'][name]=dict(completed=False,error=traceback.format_exc())
            (analysis/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        status['physical_analysis_completed']=all(x['completed'] for x in status['tasks'].values())
        (analysis/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--tag',required=True)
    a=p.parse_args();main(a.run,a.tag)
