"""Offline BKS and matched-sampling physical evaluation; partial runs stay partial."""
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
from physics.metrics import reference_vdos
from periodic import PeriodicGraphBuilder


def _spectrum_worker_init(filename,dataset_name,numbers,start,stride,block_frames,dt):
    # spawn, not fork: each worker owns its read-only HDF5 handle.
    import h5py
    global _spectral_input
    handle=h5py.File(filename,'r')
    _spectral_input=(handle,handle[dataset_name],numbers,start,stride,block_frames,dt)


def _spectrum_worker(index):
    _,dataset,numbers,start,stride,block_frames,dt=_spectral_input
    offset=index*block_frames
    a=np.asarray(dataset[start+offset*stride:start+(offset+block_frames)*stride:stride],dtype=np.float64)
    record={}
    for z in np.unique(numbers):
        frequency,power=reference_vdos(a[:,numbers==z],dt);record[str(z)]=power
    return index,frequency,record


def stats(x):
    x=np.asarray(x,dtype=float)
    blocks=[a.mean() for a in np.array_split(x,min(8,len(x))) if len(a)]
    return dict(mean=float(x.mean()),std=float(x.std()),variance=float(x.var()),
                block_mean_std=float(np.std(blocks)),blocks=len(blocks))


def paired(a,b):
    return dict(SMDP=stats(a),reference=stats(b),mean_difference=float(np.mean(a)-np.mean(b)),
                wasserstein=float(wasserstein_distance(a,b)))


def spectrum_blocks(dataset,numbers,start,frames,stride,dt,cache=None,workers=1):
    # Same 16.384 ps nonoverlapping physical windows at either cadence.
    block_frames=min(int(16384/dt),frames)
    if block_frames*dt<4096: return None
    accum={int(z):[] for z in np.unique(numbers)}
    offsets=range(0,frames-block_frames+1,block_frames)
    if cache is not None:Path(cache).mkdir(parents=True,exist_ok=True)
    if workers>1:
        if cache is None:raise ValueError('Parallel spectral analysis requires resumable cache')
        from multiprocessing import get_context
        missing=[i for i in range(len(offsets)) if not (Path(cache)/f'{i:06d}.npz').exists()]
        if missing:
            with get_context('spawn').Pool(workers,initializer=_spectrum_worker_init,
                    initargs=(dataset.file.filename,dataset.name,numbers,start,stride,block_frames,dt)) as pool:
                for completed,(index,frequency,record) in enumerate(pool.imap_unordered(_spectrum_worker,missing),1):
                    path=Path(cache)/f'{index:06d}.npz';temporary=path.with_suffix('.tmp')
                    with temporary.open('wb') as handle:np.savez(handle,frequency=frequency,**record)
                    temporary.replace(path)
                    if completed%64==0 or completed==len(missing):
                        print(f'VDOS {Path(cache).name}: {len(offsets)-len(missing)+completed}/{len(offsets)} blocks',flush=True)
        # Merge in original time order, preserving serial floating-point sums.
    for index,offset in enumerate(offsets):
        path=Path(cache)/f'{index:06d}.npz' if cache is not None else None
        if path is not None and path.exists():
            with np.load(path) as saved:
                frequency=saved['frequency']
                for z in accum:accum[z].append(saved[str(z)])
            continue
        a=np.asarray(dataset[start+offset*stride:start+(offset+block_frames)*stride:stride],dtype=np.float64)
        record={}
        for z in accum:
            frequency,power=reference_vdos(a[:,numbers==z],dt)
            accum[z].append(power)
            record[str(z)]=power
        if path is not None:
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as handle:np.savez(handle,frequency=frequency,**record)
            temporary.replace(path)
            if (index+1)%64==0 or index+1==len(offsets):
                print(f'VDOS {Path(cache).name}: {index+1}/{len(offsets)} blocks',flush=True)
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
    from analysis_tasks import main as run_tasks
    run_tasks()


if __name__=='__main__': main()
