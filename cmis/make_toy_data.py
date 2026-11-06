"""Toy two-body harmonic trajectories ONLY for checking training wiring.
Not a production molecular dynamics reference or evidence of physical fidelity.
"""
import numpy as np
from pathlib import Path


def generate(path,seed,num_samples=16,h=5,T=5):
    rng=np.random.default_rng(seed)
    mass=np.array([1.,2.],dtype=np.float32)
    z=np.array([1,8],dtype=np.int64)
    cell=np.diag([20.,20.,20.]).astype(np.float32)
    wt=2*np.pi/9.0 # internal motion period 9fs
    times=np.arange(1,T+1,dtype=np.float32)*h/T
    q0=[];v0=[];qt=[];vt=[]
    for _ in range(num_samples):
        A=rng.uniform(-0.2,0.2); B=rng.uniform(-0.2,0.2)
        rc=rng.uniform(-0.3,0.3,size=3); vc=rng.uniform(-0.02,0.02,size=3)
        def state(t):
            rel=np.array([1.0+A*np.cos(wt*t)+B*np.sin(wt*t),0,0])
            vel_rel=np.array([-A*wt*np.sin(wt*t)+B*wt*np.cos(wt*t),0,0])
            centers=rc+vc*t
            p0=centers+mass[1]/mass.sum()*rel
            p1=centers-mass[0]/mass.sum()*rel
            v0=vc+mass[1]/mass.sum()*vel_rel
            v1=vc-mass[0]/mass.sum()*vel_rel
            return np.stack([p0,p1]).astype('float32'),np.stack([v0,v1]).astype('float32')
        p,v=state(0);q0.append(p);v0.append(v)
        future=[state(float(t)) for t in times]
        qt.append(np.stack([x[0] for x in future]));vt.append(np.stack([x[1] for x in future]))
    np.savez(path,q0=np.stack(q0),v0=np.stack(v0),q_target=np.stack(qt),v_target=np.stack(vt),times_fs=times,z=z,mass=mass,cell=cell)


if __name__=='__main__':
    p=Path(__file__).parent/'toy_data';p.mkdir(exist_ok=True)
    generate(p/'toy_train.npz',42,num_samples=16)
    generate(p/'toy_val.npz',43,num_samples=4)
    print(p)