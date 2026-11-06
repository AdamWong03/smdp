import numpy as np

from physics.spcfw import verify_reference


def test_reference_verification_matches_thermo_step_and_handles_missing_forces():
    # A long trajectory saves coordinates ten times more often than thermo.
    positions=np.zeros((41,1,3));positions[:,0,0]=np.arange(41)
    raw={'positions_unwrapped_A':positions,'step':np.arange(41),
         'thermo/step':np.arange(0,41,10),
         'thermo/potential_energy_eV':np.arange(0,41,10,dtype=float)}
    class Calculator:
        def evaluate(self,q,forces=False):
            assert not forces
            return {'pe':float(q[0,0])}
    result=verify_reference(raw,Calculator(),[0,13,31,40])
    assert [r['frame'] for r in result]==[0,10,30,40]
    assert [r['thermo_index'] for r in result]==[0,1,3,4]
    assert all(r['energy_error_eV']==0 and not r['force_available'] for r in result)
