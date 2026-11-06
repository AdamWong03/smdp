"""Formal, independent-trajectory HDF5 segments, with real interior labels.

Only selected supervised states are cached. Raw reference files remain read-only.
The training reader opens each compact HDF5 lazily in the owning worker.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


COUNTS={'train':5000,'val':1250,'test':1000}


def strings(values):
    return [v.decode() if isinstance(v,bytes) else str(v) for v in values]


def selected_frames(dataset,indices):
    """Read requested frames in sequential HDF5 chunks rather than random rows."""
    unique,inverse=np.unique(indices,return_inverse=True)
    result=np.empty((len(unique),*dataset.shape[1:]),dtype=dataset.dtype)
    chunk=dataset.chunks[0] if dataset.chunks else 256
    groups=unique//chunk
    for group in np.unique(groups):
        first,last=np.searchsorted(groups,[group,group+1])
        start=int(group)*chunk
        block=dataset[start:min(start+chunk,len(dataset))]
        result[first:last]=block[unique[first:last]-start]
    return result[inverse].reshape(*indices.shape,*dataset.shape[1:])


def prepare(root,system,horizon,times,out):
    out.mkdir(parents=True,exist_ok=True)
    report={};source_sets={}
    times=np.asarray(times,dtype=float)
    for split,count in COUNTS.items():
        transition=root/system/f'transitions/{horizon:g}fs/{split}.h5'
        target=out/(split+'.h5')
        with h5py.File(transition,'r') as tr:
            dt=float(tr.attrs['source_dt_fs']);offsets=np.rint(times/dt).astype(int)
            if not np.allclose(offsets*dt,times) or times[-1]!=horizon:
                raise ValueError('Queries must be real reference frames ending at the model horizon')
            if int(tr.attrs['n_samples'])!=count or tr.attrs['split']!=split:
                raise ValueError('Unexpected formal split')
            paths=strings(tr['source_paths'][:]);source_sets[split]=set(paths)
            if not all(Path(p).parent.name.startswith(split+'_') for p in paths):
                raise ValueError('Unexpected source trajectory split')
            if target.exists():
                with h5py.File(target,'r') as old:
                    if (old.attrs.get('schema')!='cmis.real_segments.v1' or
                            old.attrs['source_transition']!=str(transition) or
                            old['q0'].shape[0]!=count or
                            not np.array_equal(old['times_fs'][:],times)):
                        raise ValueError('Existing cache differs; use a new output directory')
                report[split]={'samples':count,'cache':str(target),'reused':True}
                continue
            z=tr['atomic_numbers'][:].astype('int64');mass=tr['masses_u'][:]
            tid=tr['source_trajectory'][:].astype(int);frame=tr['source_frame'][:].astype(int)
            queries=np.r_[0,offsets]
            qs=np.empty((count,len(queries),len(z),3),dtype='f4');vs=np.empty_like(qs)
            maxerr={}
            for trajectory,path in enumerate(paths):
                samples=np.flatnonzero(tid==trajectory)
                indices=frame[samples,None]+queries[None,:]
                with h5py.File(path,'r') as raw:
                    if raw.attrs['ensemble']!='NVE' or float(raw.attrs['output_interval_fs'])!=dt:
                        raise ValueError('Source must be an NVE trajectory at the recorded cadence')
                    np.testing.assert_array_equal(raw['atomic_numbers'][:],z)
                    np.testing.assert_array_equal(raw['masses_u'][:],mass)
                    np.testing.assert_array_equal(raw['atom_ids'][:],np.arange(1,len(z)+1))
                    if indices.max()>=len(raw['time_fs']):raise IndexError('Segment exceeds source')
                    qs[samples]=selected_frames(raw['positions_unwrapped_A'],indices)
                    vs[samples]=selected_frames(raw['velocities_A_per_fs'],indices)
                    np.testing.assert_allclose(raw['cell_A'][0],raw['cell_A'][-1],atol=1e-10)
                    np.testing.assert_allclose(tr['cell_A'][samples],
                        np.broadcast_to(raw['cell_A'][0],(len(samples),3,3)),atol=1e-10)
                print(json.dumps(dict(split=split,source=path,samples=len(samples))),flush=True)
            for name,actual,expected in [
                ('initial_positions',qs[:,0],tr['positions_initial_A'][:]),
                ('final_positions',qs[:,-1],tr['positions_final_A'][:]),
                ('initial_velocities',vs[:,0],tr['velocities_initial_A_per_fs'][:]),
                ('final_velocities',vs[:,-1],tr['velocities_final_A_per_fs'][:])]:
                np.testing.assert_array_equal(actual,expected)
                maxerr[name]=float(abs(actual-expected).max())
            temporary=target.with_suffix('.tmp')
            with h5py.File(temporary,'w') as f:
                f.attrs.update(schema='cmis.real_segments.v1',system=system,split=split,
                    source_transition=str(transition),horizon_fs=horizon,source_dt_fs=dt,
                    source_paths=json.dumps(paths),position_unit='Angstrom',
                    velocity_unit='Angstrom/fs',mass_unit='u',cell_convention='row lattice vectors',
                    all_endpoints_verified=True,interior_labels='actual NVE frames; no interpolation')
                for name,a in [('q0',qs[:,0]),('v0',vs[:,0]),('q_target',qs[:,1:]),('v_target',vs[:,1:])]:
                    f.create_dataset(name,data=a,chunks=(min(32,count),*a.shape[1:]),compression='lzf')
                f['times_fs']=times;f['z']=z;f['mass']=mass;f['cell']=tr['cell_A'][:]
                f['source_trajectory']=tid;f['source_frame']=frame
            temporary.replace(target)
            report[split]=dict(samples=count,cache=str(target),source_paths=paths,max_abs_error=maxerr)
    if any(source_sets[a]&source_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Trajectory leakage across splits')
    report['protocol']=dict(system=system,horizon_fs=horizon,times_fs=times.tolist(),
                             split_trajectories_disjoint=True)
    (out/'adaptation.json').write_text(json.dumps(report,indent=2)+'\n')


class HDF5Segments(Dataset):
    def __init__(self,path):
        self.path=str(path);self.handle=None;self.pid=None
        with h5py.File(self.path,'r') as f:
            self.count,self.n_atoms,_=f['q0'].shape
            self.z=f['z'][:];self.mass=f['mass'][:];self.times=f['times_fs'][:]
            self.source_paths=json.loads(f.attrs['source_paths'])
            self.system=f.attrs['system'];self.split=f.attrs['split']
            self.horizon=float(f.attrs['horizon_fs'])
    def __len__(self):return self.count
    def __getitem__(self,index):
        if self.handle is None or self.pid!=os.getpid():
            self.handle=h5py.File(self.path,'r',rdcc_nbytes=32*1024**2);self.pid=os.getpid()
        return {name:torch.from_numpy(np.asarray(self.handle[name][index],dtype='f4').copy())
                for name in ['q0','v0','q_target','v_target','cell']}
    def scales(self):
        qsum=vsum=0.;mass=self.mass[None,None,:,None];n=0
        with h5py.File(self.path,'r') as f:
            for first in range(0,len(self),128):
                end=min(first+128,len(self));q=f['q_target'][first:end].astype(float)
                v=f['v_target'][first:end].astype(float)
                q0=f['q0'][first:end].astype(float)[:,None]
                v0=f['v0'][first:end].astype(float)[:,None]
                dq=q-q0-self.times[None,:,None,None]*v0
                dv=v-v0
                qsum+=float(np.sum(mass*dq*dq));vsum+=float(np.sum(mass*dv*dv));n+=end-first
        denominator=n*len(self.times)*3*self.mass.sum()
        return max(float(np.sqrt(qsum/denominator)),1e-7),max(float(np.sqrt(vsum/denominator)),1e-9)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--system',choices=['water','quartz'],required=True)
    p.add_argument('--horizon',type=float,required=True);p.add_argument('--times',type=float,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    prepare(a.root,a.system,a.horizon,a.times,a.out)
