"""Original reference MD, with TRAIN-only topology/verification provenance."""
from pathlib import Path

import h5py
import numpy as np

from conditional_transition.common import DATA,train_sources
from dynamics_diagnosis.reference import QuartzReference,validate_integrator
from physics.spcfw import WaterReference
from reference_md.write_lammps_protocols import bks_commands


class TrainingQuartzReference(QuartzReference):
    def __init__(self,topology,cell,numbers):
        from lammps import lammps
        self.lmp=lammps(cmdargs=['-screen','none','-log','none'])
        self.lmp.commands_list(['units real','atom_style charge','atom_modify map array',
            'boundary p p p',f'read_data "{Path(topology).resolve()}"',*bks_commands(),
            'thermo_style custom step pe','run 0'])
        self.n=int(self.lmp.get_natoms());self.cell=np.asarray(cell,dtype=float)
        lo,hi,xy,yz,xz,_,_=self.lmp.extract_box();self.origin=np.asarray(lo)
        length=np.asarray(hi)-lo
        np.testing.assert_allclose([[length[0],0,0],[xy,length[1],0],[xz,yz,length[2]]],self.cell,atol=1e-7)
        types=self.lmp.numpy.extract_atom('type')[:self.n]
        ids=self.lmp.numpy.extract_atom('id')[:self.n]
        ordered=np.empty(self.n,dtype=int);ordered[ids-1]=types
        np.testing.assert_array_equal(np.where(ordered==1,8,14),numbers)


def create_oracle(system):
    rawpath=Path(train_sources(system)[0])
    if not rawpath.parent.name.startswith('train_'):raise ValueError('Oracle verification must use train source')
    topology=DATA/system/'equilibration/snapshots'/f'{system}_1.data'
    with h5py.File(rawpath) as raw:
        cell,z,m=raw['cell_A'][0],raw['atomic_numbers'][:],raw['masses_u'][:]
        calc=(TrainingQuartzReference(topology,cell,z) if system=='quartz'
              else WaterReference(topology,cell,z))
        verification=validate_integrator(system,raw,calc)
    verification.update(reference_for_training=True,verification_source=str(rawpath),
        topology=str(topology),purpose='training-label generation only; never inference')
    if not verification['passed']:
        calc.close();raise RuntimeError(f'Train reference verification failed: {verification}')
    return calc,cell,z,m,verification
