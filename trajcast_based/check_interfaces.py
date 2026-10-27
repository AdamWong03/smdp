"""Numerical checks of the untouched upstream input/output conventions."""
import json
import ase.io
import numpy as np
import torch
from torch_geometric.data import Batch
from trajcast_based.acquire import RUN
from trajcast_based.official import load


def main():
    from trajcast.data.atomic_graph import AtomicGraph
    from trajcast.utils.misc import convert_ase_atoms_to_dictionary, format_values_in_dictionary
    model,_,_=load();model.eval()
    atoms=ase.io.read(RUN/'official/datasets/quartz/test.extxyz',index=':2')
    def graph(a):
        return AtomicGraph.from_atoms_dict(format_values_in_dictionary(convert_ase_atoms_to_dictionary(a)),4.5,{8:0,14:1})
    graphs=[graph(a) for a in atoms]
    errors={}
    with torch.no_grad():
        single=[model(Batch.from_data_list([g.clone()])).target for g in graphs]
        combined=model(Batch.from_data_list([g.clone() for g in graphs])).target
        errors['batch_single_max_normalized']=float((combined-torch.cat(single)).abs().max())
        torch.testing.assert_close(combined,torch.cat(single),atol=1e-10,rtol=1e-9)
        for i,g in enumerate(graphs):
            # The two output vector fields are normalized displacement and final velocity.
            p=single[i][:,3:]*model.rms_targets[1]
            d=single[i][:,:3]*model.rms_targets[0]
            errors[f'net_velocity_momentum_{i}']=float((p*g.atomic_masses).sum(0).abs().max())
            errors[f'net_displacement_mass_{i}']=float((d*g.atomic_masses).sum(0).abs().max())
            assert errors[f'net_velocity_momentum_{i}']<1e-9
        # Reflection/rotation simultaneously transforms position, cell, velocity and targets.
        rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        a=atoms[0].copy();a.positions=a.positions@rot.numpy().T
        a.set_cell(a.cell.array@rot.numpy().T)
        for k in ['velocities','displacements','update_velocities']:a.arrays[k]=a.arrays[k]@rot.numpy().T
        pred=model(Batch.from_data_list([graph(a)])).target.reshape(-1,2,3)
        expected=single[0].reshape(-1,2,3)@rot.T
        errors['O3_max_normalized']=float((pred-expected).abs().max())
        torch.testing.assert_close(pred,expected,atol=1e-9,rtol=1e-8)
        a=atoms[0].copy();a.positions[::2]+=a.cell.array[0]
        p=model(Batch.from_data_list([graph(a)])).target
        errors['periodic_reimage_max_normalized']=float((p-single[0]).abs().max())
        torch.testing.assert_close(p,single[0],atol=1e-9,rtol=1e-8)
    errors.update(passed=True,precision=str(torch.get_default_dtype()),
        node_irreps_after_last_mp=str(model._int_out.mp_layer_3.irreps_out['node_features']),
        adapter_insertion='EfficientTrajCastModel._int_out: after mp_layer_3, before LinearCompression',
        official_cell_A=atoms[0].cell.array.tolist(),official_masses=sorted(set(atoms[0].get_masses())))
    (RUN/'interface_checks.json').write_text(json.dumps(errors,indent=2)+'\n')
    print(json.dumps(errors,indent=2))


if __name__=='__main__':main()
