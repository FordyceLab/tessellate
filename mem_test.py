from tesselate.data import TesselateDataset, make_sparse_mat
from torch.utils.data import DataLoader
from tqdm import *
import gc
import time
import torch.multiprocessing as multiprocessing

def dict_collate(batch):
    return batch[0]

if __name__ == '__main__':
    data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/success/training_30_ids.txt', 'data/contacts.hdf5', 'complete')
    dataloader = DataLoader(data, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False,
                            collate_fn=dict_collate)
    
    dl = iter(dataloader)
    
    i = 0
    
    for sample in tqdm(dl):
        i+=1
        if i >= 40:
            break
    