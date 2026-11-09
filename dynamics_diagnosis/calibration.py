"""Quantify the real-reference continuation/serialization floor on all probes."""
import h5py
import numpy as np
import pandas as pd

from dynamics_diagnosis.common import ARTIFACTS,REPORT,source,initial_sources,save
from dynamics_diagnosis.reference import calculator
from dynamics_diagnosis.experiment import ke


def main():
    rows=[]
    for system in ['quartz','water']:
        with h5py.File(source(system)) as raw:
            calc=calculator(system,raw)
            m=raw['masses_u'][:]
        paths=initial_sources(system)
        handles=[h5py.File(p) for p,_ in paths]
        try:
            with h5py.File(ARTIFACTS/system/'bank_reference.h5') as bank:
                for i in range(len(bank['q'])):
                    replica,frame=int(bank['replica'][i]),int(bank['frame'][i])
                    raw=handles[replica]
                    lag=round(float(bank.attrs['horizon_fs'])/float(raw.attrs['output_interval_fs']))
                    stored_q=raw['positions_unwrapped_A'][frame+lag].astype(float)
                    stored_v=raw['velocities_A_per_fs'][frame+lag].astype(float)
                    fresh_q,fresh_v=bank['true_q'][i],bank['true_v'][i]
                    energy_offset=(float(bank['true_PE'][i])+ke(fresh_v,m)-
                                   calc.evaluate(stored_q)['pe']-ke(stored_v,m))
                    rows.append(dict(system=system,replica=replica,frame=frame,
                        q_RMSE_A=float(np.sqrt(np.mean((fresh_q-stored_q)**2))),
                        v_RMSE_A_fs=float(np.sqrt(np.mean((fresh_v-stored_v)**2))),
                        fresh_minus_stored_energy_eV=float(energy_offset)))
        finally:
            calc.close()
            for f in handles:f.close()
    df=pd.DataFrame(rows)
    df.to_csv(REPORT/'reference_continuation_floor.csv',index=False)
    result={system:dict(samples=len(g),mean_q_RMSE_A=float(g.q_RMSE_A.mean()),
        mean_v_RMSE_A_fs=float(g.v_RMSE_A_fs.mean()),
        mean_energy_offset_eV=float(g.fresh_minus_stored_energy_eV.mean()),
        std_energy_offset_eV=float(g.fresh_minus_stored_energy_eV.std(ddof=0)))
        for system,g in df.groupby('system')}
    save(REPORT/'reference_continuation_floor.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
