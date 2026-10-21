"""Offline BKS and matched-sampling physical evaluation; partial runs stay partial."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physics.bks import make_calculator, energy
from physics.units import KINETIC_TO_EV, KB_EV_PER_K
from physics.metrics import reference_vdos
from periodic import PeriodicGraphBuilder


def stats(x):
    x=np.asarray(x,dtype=float)
    blocks=[a.mean() for a in np.array_split(x,min(8,len(x))) if len(a)]
    return dict(mean=float(x.mean()),std=float(x.std()),variance=float(x.var()),
                block_mean_std=float(np.std(blocks)),blocks=len(blocks))


def paired(a,b):
    return dict(SMDP=stats(a),reference=stats(b),mean_difference=float(np.mean(a)-np.mean(b)),
                wasserstein=float(wasserstein_distance(a,b)))


def spectrum_blocks(dataset,numbers,start,frames,stride,dt):
    # Same 16.384 ps nonoverlapping physical windows at either cadence.
    block_frames=min(int(16384/dt),frames)
    if block_frames*dt<4096: return None
    accum={int(z):[] for z in np.unique(numbers)}
    for offset in range(0,frames-block_frames+1,block_frames):
        a=np.asarray(dataset[start+offset*stride:start+(offset+block_frames)*stride:stride],dtype=np.float64)
        for z in accum:
            frequency,power=reference_vdos(a[:,numbers==z],dt)
            accum[z].append(power)
    return frequency,{z:np.asarray(x) for z,x in accum.items()}


def rdf_sample(frames,cell,z,first,second,rmax,bins=180):
    builder=PeriodicGraphBuilder(rmax); edges=np.linspace(0,rmax,bins+1);hist=[]
    for q in frames:
        e,s=builder.one(q,cell)
        mask=(z[e[1]]==first)&(z[e[0]]==second)
        d=np.linalg.norm(q[e[0,mask]]+s[mask]-q[e[1,mask]],axis=1)
        hist.append(np.histogram(d,edges)[0])
    shell=4*np.pi/3*np.diff(edges**3); volume=abs(np.linalg.det(cell))
    denominator=np.sum(z==first)*(np.sum(z==second)-(first==second))/volume*shell
    return .5*(edges[:-1]+edges[1:]),np.asarray(hist)/np.maximum(denominator,1e-30)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.candidate) as c,h5py.File(args.reference) as r:
        completed=int(c.attrs['completed_steps']);thermal=int(c.attrs.get('thermalization_steps',0))
        production=completed>thermal
        start=thermal if production else 0
        frames=completed-start+1
        if frames<2:
            (args.output/'metrics.json').write_text(json.dumps({'available':False,'reason':'no complete trajectory step'}));return
        z=c['atomic_numbers'][:];m=c['masses_u'][:];cell=c['cell'][:]
        dt=16.;n=len(m);dof=3*n-3
        common=min(frames,(len(r['time_fs'])-1)//16+1)
        result=dict(candidate=str(args.candidate),reference=str(args.reference),stable=bool(c.attrs['stable']),
                    failure=str(c.attrs['failure_reason']),production_available=production,
                    evaluation_phase='production' if production else 'failed_thermalization_diagnostic',
                    duration_ps=(common-1)*.016,requested_production_ps=int(c.attrs['requested_steps'])*.016,
                    minimum_distance_A=float(c['minimum_distance_A'][:].min()))
        # Uniform physical-time samples; reference thermodynamics uses its actual
        # timestamps (not the coordinate-frame index: thermo cadence differs).
        idx=np.unique(np.linspace(0,common-1,min(2000,common),dtype=int));times=idx*16.
        ci=start+idx
        cv=np.asarray(c['velocities'][ci],dtype=float);rv=np.asarray(r['velocities_A_per_fs'][idx*16],dtype=float)
        cke=.5*KINETIC_TO_EV*np.sum(m[None,:,None]*cv**2,axis=(1,2))
        rke=.5*KINETIC_TO_EV*np.sum(m[None,:,None]*rv**2,axis=(1,2))
        result['temperature_K']=paired(2*cke/(dof*KB_EV_PER_K),2*rke/(dof*KB_EV_PER_K))
        result['kinetic_energy_eV']=paired(cke,rke)
        result['element_kinetic_energy_eV']={}
        for element in np.unique(z):
            mask=z==element
            a=.5*KINETIC_TO_EV*np.sum(m[None,mask,None]*cv[:,mask]**2,axis=(1,2))
            b=.5*KINETIC_TO_EV*np.sum(m[None,mask,None]*rv[:,mask]**2,axis=(1,2))
            result['element_kinetic_energy_eV'][str(element)]=paired(a,b)
        calc=make_calculator();cpe=[];rpe=[]
        for frame, ref_frame in zip(ci,idx*16):
            atoms=Atoms(numbers=z,positions=c['positions'][int(frame)],cell=cell,pbc=True)
            cpe.append(energy(atoms,calc)[1]) # same raw coul/dsf zero as formal MD
            atoms=Atoms(numbers=z,positions=r['positions_unwrapped_A'][int(ref_frame)],cell=cell,pbc=True)
            rpe.append(energy(atoms,calc)[1])
        cpe=np.asarray(cpe)
        rpe=np.asarray(rpe)
        result['energy_evaluation']='Identical offline BKS on exact sampled configurations for candidate and reference; raw coul/dsf convention'
        result['potential_energy_eV']=paired(cpe,rpe)
        total=cpe+cke;reference_total=rpe+rke
        result['total_energy_drift_meV_atom_ps']=dict(SMDP=float(np.polyfit(times/1000,total,1)[0]*1000/n),
                                                   reference=float(np.polyfit(times/1000,reference_total,1)[0]*1000/n))
        result['energy_change_eV']=dict(SMDP=float(total[-1]-total[0]),reference=float(reference_total[-1]-reference_total[0]))
        pd.DataFrame(dict(time_fs=times,SMDP_PE_eV=cpe,reference_PE_eV=rpe,SMDP_total_eV=total,reference_total_eV=reference_total,
                          SMDP_temperature_K=2*cke/(dof*KB_EV_PER_K),reference_temperature_K=2*rke/(dof*KB_EV_PER_K))).to_csv(args.output/'thermodynamics.csv',index=False)
        if c.attrs['ensemble']=='nve':
            cq=c['positions'][ci];rq=r['positions_unwrapped_A'][idx*16]
            qr=np.sqrt(np.mean((cq-rq)**2,axis=(1,2)));vr=np.sqrt(np.mean((cv-rv)**2,axis=(1,2)))
            pd.DataFrame(dict(time_fs=times,position_RMSE_A=qr,velocity_RMSE_A_fs=vr)).to_csv(args.output/'nve_trajectory_error.csv',index=False)
        result['segment_join_derivative_RMS_A_fs']=stats(c['segment_join_derivative_RMS_A_fs'][start:completed+1])
        result['VDOS']={}
        for label,cd,cs,cf,rs,dt in [('macro_16fs',c['velocities'],start,common,16,16.),
                                    ('queried_1fs',c['dense/velocities'],start*16,(common-1)*16+1,1,1.)]:
            cand=spectrum_blocks(cd,z,cs,cf,1,dt)
            ref=spectrum_blocks(r['velocities_A_per_fs'],z,0,cf,rs,dt)
            if cand is None or ref is None:
                result['VDOS'][label]={'available':False,'reason':'less than 4.096 ps; no reliable spectral estimate'};continue
            freq,cp=cand;_,rp=ref
            result['VDOS'][label]={'nyquist_THz':float(freq[-1]),'elements':{}}
            fig,axes=plt.subplots(2,1,figsize=(7,6));rows=[]
            for ax,element in zip(axes,sorted(cp)):
                a,b=cp[element].mean(0),rp[element].mean(0)
                overlaps=np.minimum(cp[element],rp[element]).sum(1)
                result['VDOS'][label]['elements'][str(element)]=dict(overlap=float(np.minimum(a,b).sum()),block_overlap=stats(overlaps),
                    reference_split_half_overlap=float(np.minimum(rp[element][::2].mean(0),rp[element][1::2].mean(0)).sum()) if len(rp[element])>1 else None)
                ax.plot(freq,b,label='BKS');ax.plot(freq,a,label='SMDP');ax.set_title(f'Z={element}, {label}')
                ax.set_xlim(0,min(float(freq[-1]),80));ax.legend();ax.set_xlabel('THz')
                rows.extend(dict(element=element,frequency_THz=float(f),reference=float(x),SMDP=float(y)) for f,x,y in zip(freq,b,a))
            fig.tight_layout();fig.savefig(args.output/f'vdos_{label}.png',dpi=160);plt.close(fig)
            pd.DataFrame(rows).to_csv(args.output/f'vdos_{label}.csv',index=False)
        ri=np.unique(np.linspace(0,common-1,min(512,common),dtype=int))
        cq=c['positions'][start+ri];rq=r['positions_unwrapped_A'][ri*16]
        rmax=min(6.,.49/np.linalg.norm(np.linalg.inv(cell),axis=0).max())
        result['RDF']={};rows=[]
        fig,axes=plt.subplots(2,1,figsize=(7,6))
        for ax,(a,b) in zip(axes,[(14,8),(8,8)]):
            radius,cr=rdf_sample(cq,cell,z,a,b,rmax);_,rr=rdf_sample(rq,cell,z,a,b,rmax)
            cm,rm=cr.mean(0),rr.mean(0)
            # First-shell peak, not an arbitrary global maximum at later shells.
            mask=(radius>1)&(radius<2.2) if a==14 else (radius>2)&(radius<3.2)
            choices=np.flatnonzero(mask);ic=choices[np.argmax(cm[mask])];ir=choices[np.argmax(rm[mask])]
            result['RDF'][f'{a}-{b}']=dict(RMSE=float(np.sqrt(np.mean((cm-rm)**2))),
                peak_position_A=dict(SMDP=float(radius[ic]),reference=float(radius[ir]),difference=float(radius[ic]-radius[ir])),
                peak_height=dict(SMDP=float(cm[ic]),reference=float(rm[ir]),difference=float(cm[ic]-rm[ir])),
                block_curve_RMSE=stats([np.sqrt(np.mean((x.mean(0)-y.mean(0))**2)) for x,y in zip(np.array_split(cr,min(8,len(cr))),np.array_split(rr,min(8,len(rr))))]))
            ax.plot(radius,rm,label='BKS');ax.plot(radius,cm,label='SMDP');ax.legend();ax.set_title(f'{a}-{b}');ax.set_xlabel('Angstrom')
            rows.extend(dict(pair=f'{a}-{b}',r_A=float(r),reference=float(x),SMDP=float(y)) for r,x,y in zip(radius,rm,cm))
        fig.tight_layout();fig.savefig(args.output/'rdf.png',dpi=160);plt.close(fig)
        pd.DataFrame(rows).to_csv(args.output/'rdf.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(times/1000,total-total[0],label='SMDP');axes[0].plot(times/1000,reference_total-reference_total[0],label='BKS')
        axes[0].set(xlabel='ps',ylabel='Total energy change (eV)');axes[0].legend()
        axes[1].hist(cpe,bins=40,alpha=.5,density=True,label='SMDP');axes[1].hist(rpe,bins=40,alpha=.5,density=True,label='BKS');axes[1].legend()
        fig.tight_layout();fig.savefig(args.output/'energy.png',dpi=160);plt.close(fig)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__': main()
