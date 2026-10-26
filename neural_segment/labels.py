"""BKS training labels only. No import from this module in neural inference."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from physics.bks import make_calculator, evaluate, set_bks_charges
from physics.units import FORCE_TO_ACCEL
from reference_data import QuartzSegments

OFFSETS=np.array([0,4,8,12,16])


def atoms_state(q,v,z,m,cell,calc):
    atoms=Atoms(numbers=z,positions=q,masses=m,cell=cell,pbc=True)
    atoms.set_velocities(v/units.fs)
    set_bks_charges(atoms);atoms.calc=calc
    return atoms


def short_reference(q,v,z,m,cell,calc):
    """ASE's mature velocity-Verlet, 1 fs, force/energy from existing LAMMPS BKS."""
    atoms=atoms_state(q,v,z,m,cell,calc)
    dynamics=VelocityVerlet(atoms,timestep=units.fs,logfile=None)
    qs=[];vs=[];acc=[];energy=[]
    for t in range(17):
        _,force,pe=evaluate(atoms,calc)
        qs.append(atoms.positions.copy());vs.append(atoms.get_velocities()*units.fs)
        energy.append(pe)
        if t in OFFSETS:acc.append(force/m[:,None]*FORCE_TO_ACCEL)
        if t<16:dynamics.run(1)
    return np.array(qs),np.array(vs),np.array(acc),np.array(energy)


def verify(root,output):
    """Independent checks before using any generated acceleration supervision."""
    report={};calc=make_calculator()
    path=root/'quartz/raw_nve/train_01/trajectory.h5'
    with h5py.File(path) as f:
        z,m=f['atomic_numbers'][:],f['masses_u'][:]
        errs=[];relative=[];vel=[];energies=[]
        for frame in [100,1000,10000]:
            q=f['positions_unwrapped_A'][frame].astype(float);v=f['velocities_A_per_fs'][frame].astype(float);cell=f['cell_A'][frame]
            atoms=atoms_state(q,v,z,m,cell,calc)
            _,force,pe=evaluate(atoms,calc)
            original=f['forces_eV_per_A'][frame]
            errs.append(np.max(abs(force-original)))
            relative.append(np.linalg.norm(force-original)/np.linalg.norm(original))
            dv=v-f['velocities_A_per_fs'][frame-1].astype(float)
            expected=.5*(original.astype(float)+f['forces_eV_per_A'][frame-1].astype(float))/m[:,None]*FORCE_TO_ACCEL
            vel.append(np.linalg.norm(dv-expected)/np.linalg.norm(dv))
            # Force as the derivative of the same raw BKS potential.
            eps=1e-5;direction=np.zeros_like(q);direction[0,0]=eps
            atoms.positions=q+direction;plus=evaluate(atoms,calc)[2]
            atoms.positions=q-direction;minus=evaluate(atoms,calc)[2]
            energies.append(abs((plus-minus)/(2*eps)+force[0,0]))
        frame=100
        ref=short_reference(f['positions_unwrapped_A'][frame],f['velocities_A_per_fs'][frame],
                            z,m,f['cell_A'][frame],calc)
        qerr=float(np.sqrt(np.mean((ref[0]-f['positions_unwrapped_A'][frame:frame+17])**2)))
        verr=float(np.sqrt(np.mean((ref[1]-f['velocities_A_per_fs'][frame:frame+17])**2)))
    calc.lmp.close()
    report=dict(source=str(path),force_max_error_eV_A=float(max(errs)),
        force_relative_RMS=float(max(relative)),
        stored_velocity_verlet_relative_RMS=float(max(vel)),
        force_vs_energy_gradient_max_error_eV_A=float(max(energies)),
        regenerated_16fs_position_RMSE_A=qerr,regenerated_velocity_RMSE_A_fs=verr,
        acceleration_conversion=FORCE_TO_ACCEL,integrator='ASE velocity-Verlet, 1 fs; existing LAMMPS BKS forces',
        reference_precision_note='Original dump uses ~6 significant digits; positions/forces stored float32. Recomputed force tolerance 0.1% RMS accounts for coordinate rounding, not potential changes.',
        thresholds=dict(force_relative=.001,velocity_verlet_relative=.0001,energy_gradient=1e-4,q=3e-5,v=2e-6))
    report['passed']=bool(max(relative)<.001 and max(vel)<.0001 and max(energies)<1e-4 and qerr<3e-5 and verr<2e-6)
    (output/'bks_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('BKS labels failed consistency checks; no training')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    verify(args.data_root,args.output)
    data=QuartzSegments(args.data_root,'train',args.output/'cache')
    path=args.output/'original_acceleration.h5'
    source=args.data_root/'quartz/transitions/16fs/train.h5'
    identity=hashlib.sha256(source.read_bytes()).hexdigest()
    calc=make_calculator()
    with h5py.File(path,'a') as out:
        if 'a' not in out:
            out.create_dataset('a',shape=(5000,5,162,3),dtype='f4',chunks=(1,5,162,3))
            out.create_dataset('potential_eV',shape=(5000,5),dtype='f8')
            out.create_dataset('done',data=np.zeros(5000,dtype=bool))
            out.attrs.update(source_sha256=identity,offsets_fs=OFFSETS,split='train')
        assert out.attrs['source_sha256']==identity
        # Evaluate actual raw coordinates, not interpolated or rounded endpoints.
        for tid,raw_path in enumerate(data.sources['paths']):
            selected=np.flatnonzero(np.array(data.sources['trajectories'])==tid)
            with h5py.File(raw_path) as raw:
                for idx in selected:
                    if out['done'][idx]:continue
                    frame=data.sources['frames'][idx];a=[];pe=[]
                    for off in OFFSETS:
                        atoms=atoms_state(raw['positions_unwrapped_A'][frame+off],
                            raw['velocities_A_per_fs'][frame+off],data.z.numpy(),data.m.numpy(),
                            raw['cell_A'][frame+off],calc)
                        _,force,energy=evaluate(atoms,calc)
                        a.append(force/data.m.numpy()[:,None]*FORCE_TO_ACCEL);pe.append(energy)
                    out['a'][idx]=a;out['potential_eV'][idx]=pe;out['done'][idx]=True
                    if idx%100==0:
                        out.flush();print(json.dumps(dict(labels_complete=int(out['done'][:].sum()))),flush=True)
        out.attrs['completed']=bool(out['done'][:].all())
        a=out['a'][:];scale=float(np.sqrt(np.mean(a.astype(float)**2)))
    calc.lmp.close()
    (args.output/'acceleration_scale.json').write_text(json.dumps(dict(acceleration_rms=scale,training_samples=5000,offsets_fs=OFFSETS.tolist()),indent=2))


if __name__=='__main__':main()
