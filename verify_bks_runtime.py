"""Verify restored runtime against published R1 fixed snapshots; no MD integration."""
import argparse, ctypes, json, subprocess, os, sys
import lammps
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from physics.bks import make_calculator, energy


def main():
    p=argparse.ArgumentParser();p.add_argument('--r1',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lib=ctypes.CDLL('libmpi.so.12');buf=ctypes.create_string_buffer(8192);n=ctypes.c_int()
    lib.MPI_Get_library_version(buf,ctypes.byref(n))
    rows=[];calc=make_calculator()
    for name in ['nve_test_09','nve_test_10','nvt']:
        ref=a.data_root/('quartz/physical_reference/nvt_15ns/production.h5' if name=='nvt' else f'quartz/raw_nve/{name[4:]}/trajectory.h5')
        table=pd.read_csv(a.r1/f'{name}_analysis/thermodynamics.csv')
        with h5py.File(a.r1/(name+'.h5')) as c,h5py.File(ref) as r:
            start=int(c.attrs.get('thermalization_steps',0))
            for i in [0,1,len(table)//2,len(table)-1]:
                row=table.iloc[i];t=int(row.time_fs);frame=start+t//16
                for label,q in [('SMDP',c['positions'][frame]),('reference',r['positions_unwrapped_A'][t])]:
                    atoms=Atoms(numbers=c['atomic_numbers'][:],positions=q,cell=c['cell'][:],pbc=True)
                    observed=energy(atoms,calc)[1];expected=float(row[label+'_PE_eV'])
                    rows.append(dict(run=name,time_fs=t,source=label,expected_eV=expected,observed_eV=observed,error_eV=observed-expected))
    maximum=max(abs(x['error_eV']) for x in rows)
    shared=Path(lammps.__file__).parent/'liblammps.so'
    result=dict(passed=maximum<1e-7,max_abs_error_eV=maximum,tolerance_eV=1e-7,mpi_version=buf.value.decode(),
                lammps_version=calc.lmp.version(),samples=rows,python=sys.executable,
                library_path=os.environ.get('LD_LIBRARY_PATH',''),lammps_library=str(shared),
                ldd=subprocess.run(['ldd',str(shared)],capture_output=True,text=True,check=True).stdout,
                readelf_dynamic=subprocess.run(['readelf','-d',str(shared)],capture_output=True,text=True,check=True).stdout)
    calc.lmp.close()
    (a.output/'bks_runtime_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']: raise RuntimeError('BKS parity failed: do not proceed with new energy analysis')


if __name__=='__main__': main()
