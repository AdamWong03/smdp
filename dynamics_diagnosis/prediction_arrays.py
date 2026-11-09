"""Retain full same-state q/v predictions behind compact diagnostic CSVs."""
import h5py
import numpy as np

from dynamics_diagnosis.common import ARTIFACTS,checkpoints
from dynamics_diagnosis.experiment import predict
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational


def main():
    for system in ['quartz','water']:
        for name,checkpoint in checkpoints(system).items():
            directory=ARTIFACTS/system/name
            if not (directory/'completed.json').exists():continue
            model,_=load_relational(checkpoint)
            builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
            with h5py.File(directory/'validation_50ps.h5') as f:
                cell,z,m=f['cell'][:],f['z'][:],f['mass'][:]
            for bank_name in dict.fromkeys(['reference',name,'k3_stage2']):
                path=directory/f'predictions_{bank_name}.h5'
                if path.exists():continue
                source=ARTIFACTS/system/f'bank_{bank_name}.h5'
                with h5py.File(source) as bank,h5py.File(path,'x') as f:
                    f.attrs.update(purpose='evaluation only; never training',checkpoint=str(checkpoint),
                        state_bank=str(source),true_q_v_in_source_bank=True)
                    qout=f.create_dataset('predicted_q',bank['q'].shape,dtype='f4')
                    vout=f.create_dataset('predicted_v',bank['v'].shape,dtype='f4')
                    dqout=f.create_dataset('predicted_dq',bank['q'].shape,dtype='f8')
                    dvout=f.create_dataset('predicted_dv',bank['v'].shape,dtype='f8')
                    for start in range(0,len(bank['q']),8):
                        q,v=bank['q'][start:start+8],bank['v'][start:start+8]
                        pq,pv=predict(model,q,v,cell,z,m,builder)
                        qout[start:start+8],vout[start:start+8]=pq,pv
                        dqout[start:start+8],dvout[start:start+8]=pq-q,pv-v
                    f.attrs['completed']=True
            del model


if __name__=='__main__':main()
