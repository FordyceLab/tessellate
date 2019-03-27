import matplotlib.pyplot as plt
import numpy as np
import sys
from tesselate.data import TesselateDataset, make_sparse_mat
from tesselate.model import Network
import torch
import torch.multiprocessing as multiprocessing 
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import *
import time


def plot_channels(pdb_id, target, out, epoch):
    
    cont_chan = (
        'vdw',
        'proximal',
        'hydrogen_bond',
        'weak_hydrogen_bond',
        'halogen_bond',
        'ionic',
        'metal_complex',
        'aromatic',
        'hydrophobic',
        'carbonyl',
        'polar',
        'weak_polar',
    )
    
    fig, ax = plt.subplots(target.shape[0],
                           3,
                           figsize=(2 * 3, 2 * target.shape[0]))
    
    fig.suptitle('{} - Epoch: {:06d}'.format(pdb_id, epoch), fontsize=16)

    for i in range(0, target.shape[0] * 3, 3):
        j = int(i / 3)

        ax1 = plt.subplot(target.shape[0], 3, i + 1)
        ax1.imshow(out[j], cmap='gray', vmin=0, vmax=1)
        ax2 = plt.subplot(target.shape[0], 3, i + 2)
        ax2.imshow(out[j].round(0), cmap='gray', vmin=0, vmax=1)
        
        ax2.annotate(cont_chan[j], xy=(0.5, 1), xytext=(0, 10),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
        
        ax3 = plt.subplot(target.shape[0], 3, i + 3)
        ax3.imshow(target[j], cmap='gray', vmin=0, vmax=1)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.subplots_adjust(hspace=0.75)
    
    plt.savefig('figs/{}_{:06d}.png'.format(pdb_id, epoch))
    plt.close()
    
    
def plot_worker(queue):
    example = queue.get()
    while example != 'plot_complete':
        if len(example) == 4:
            plot_channels(*example)
        
        example = queue.get()
    


if __name__ == '__main__':
    
#     multiprocessing.set_start_method('spawn')
#     queue = mp.Queue()
#     p = mp.Pool(10, plot_worker, (queue,))

    INPUT_SIZE = 25
    GRAPH_CONV = 7
    FEED = 'single_chain'
    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')
    cpu = torch.device('cpu')

    model = Network(INPUT_SIZE, 7, cuda0, cuda1)

    data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/training_30_ids.txt', 'data/contacts.hdf5', FEED)
    dataloader = DataLoader(data, batch_size=1, shuffle=True,
                            num_workers=38, pin_memory=True,
                            collate_fn=lambda b: b[0])

    opt = optim.SGD(model.parameters(), lr = .01, momentum=0.9) #, weight_decay=1e-4)

    for epoch in trange(10000):

        total_loss = 0

        for sample in tqdm(dataloader):
            for idx in tqdm(range(len(sample['id'])), leave=False):

                start = time.time()

                opt.zero_grad()
                
                atomtypes = torch.from_numpy(sample['atomtypes'][idx][:, 3])
                adjacency = make_sparse_mat(sample['adjacency'][idx], 'adj')
                memberships = make_sparse_mat(sample['memberships'][idx], 'mem')
                target = make_sparse_mat(sample['target'][idx], 'tar', int(np.max(sample['memberships'][idx][:, 0]) + 1)).to_dense()

                    
                adjacency = adjacency.float().to(cuda0)
                memberships = memberships.float().to(cuda0)
                target = target.float().to(cuda1)
                
#                 print(adjacency.shape)
#                 print(atomtypes.shape)
#                 print(memberships.shape)
#                 print(target.shape)

                out = model(adjacency, atomtypes, memberships)

                loss = F.binary_cross_entropy(out, target, reduction='mean')

            #     print(torch.sum(target))

                loss = F.binary_cross_entropy(out, target)

#                 print(torch.sum(loss * target), torch.sum(target), torch.sum(loss * target) / torch.sum(target))

                loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * ((target - 1) + 2))  / torch.sum((target - 1) + 2)


                back_start = time.time()

                loss.backward()

                back_end = time.time()
#                     print('Backward: {}'.format(back_end - back_start))

                update_start = time.time()

                opt.step()

                update_end = time.time()
#                     print('Update: {}'.format(update_end - update_start))


                total_loss += loss.data

                pdb_id = sample['id'][idx]
                out = out.data.to(cpu).numpy()
                target = target.to(cpu).numpy()

#                 if epoch % 50 == 0:
#                     queue.put((pdb_id, target, out, epoch))

#                     del (adjacency, 
#                          atomtypes, 
#                          memberships, 
#                          target,
#                          out,
#                          loss)

#                     torch.cuda.empty_cache()

                end = time.time()

#                     print('{}: {}'.format(sample['id'][idx][0], end - start))


#         if epoch % 10 == 0:
        print('Epoch: {}, Loss: {:02f}'.format(epoch, float(total_loss) / 10))
    
    queue.put('plot_complete')
    queue.close()
    queue.join_thread()
    p.join()
                