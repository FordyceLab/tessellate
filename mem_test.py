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
    dataloader = DataLoader(data, batch_size=1, shuffle=True,
                            num_workers=500, pin_memory=True,
                            collate_fn=dict_collate)
    
    dl = iter(dataloader)
    
    for sample in dl:
        del sample
        tqdm.write('Queue size: {}'.format(dl.data_queue.qsize()))
        continue
    