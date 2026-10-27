"""Read every exported sample through the actual official Dataset/DataLoader."""
import json
import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast_based.acquire import ROOT,RUN


def main():
    torch.set_default_dtype(torch.float64)
    from trajcast.data.dataset import AtomicGraphDataset
    report={}
    for split in ['train','val','test']:
        dataset=AtomicGraphDataset(root=str(RUN/'independent16'),name=f'quartz16_{split}',
            cutoff_radius=4.5,files=[],atom_type_mapper={8:0,14:1})
        maxerr=0.;norm=np.zeros(2);idx=0
        with h5py.File(ROOT/f'data/smdp_reference_md_v1/quartz/transitions/16fs/{split}.h5') as f:
            for batch in DataLoader(dataset,batch_size=2,shuffle=False):
                assert not (batch.batch[batch.edge_index[0]]!=batch.batch[batch.edge_index[1]]).any()
                for g in batch.to_data_list():
                    for actual,expected in [
                        (g.displacements.numpy(),f['positions_final_A'][idx].astype('float64')-f['positions_initial_A'][idx].astype('float64')),
                        (g.velocities.numpy(),f['velocities_initial_A_per_fs'][idx]),
                        (g.update_velocities.numpy(),f['velocities_final_A_per_fs'][idx]),
                        (g.atomic_masses.numpy().ravel(),f['masses_u'][:]),
                        (g.cell.numpy().reshape(3,3),f['cell_A'][idx])]:
                        err=float(np.max(np.abs(actual-expected))); maxerr=max(maxerr,err)
                        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                    norm += [float(g.displacements.square().mean()),float(g.update_velocities.square().mean())]
                    idx+=1
        assert idx==len(dataset)
        report[split]=dict(samples=idx,roundtrip_max_abs_error=maxerr,no_cross_graph_edges=True)
        if split=='train': report['training_only_rms']=np.sqrt(norm/idx).tolist()
    (RUN/'independent16/cache_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
