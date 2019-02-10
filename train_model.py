import matplotlib.pyplot as plt
import numpy as np
from tesselate.data import TesselateDataset
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


def plot_channels(pdb_id, target, out):
    plt.subplots(target.shape[0],
                 3,
                 figsize=(18, 7 * target.shape[0]))

    for i in range(0, target.shape[0] * 3, 3):
        j = int(i / 3)

        ax1 = plt.subplot(target.shape[0], 3, i + 1)
        ax1.imshow(out[j], cmap='gray')
        ax2 = plt.subplot(target.shape[0], 3, i + 2)
        ax2.imshow(out[j].round(0), cmap='gray')
        ax3 = plt.subplot(target.shape[0], 3, i + 3)
        ax3.imshow(target[j], cmap='gray')

    plt.savefig('figs/{}_{:06d}.png'.format(sample['id'][0], epoch))
    plt.close()


if __name__ == '__main__':
    
#     multiprocessing.set_start_method('spawn')
    with mp.Pool(processes=5) as pool:

        INPUT_SIZE = 10
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        cpu = torch.device('cpu')

        model = Network(INPUT_SIZE, 10, cuda0, cuda1)

        data = TesselateDataset('pdb_ids.txt', 'data/contacts.hdf5')
        dataloader = DataLoader(data, batch_size=1, shuffle=True,
                                num_workers=25, pin_memory=True)

        opt = optim.SGD(model.parameters(), lr = .01, momentum=0.9, weight_decay=1e-4)

        for epoch in trange(10):

            total_loss = 0

            for sample in tqdm(dataloader):

                start = time.time()

                opt.zero_grad()

                adjacency = sample['adjacency'].squeeze().to_sparse().to(cuda0)
                atomtypes = sample['atomtypes'].squeeze()
                memberships = sample['memberships'].squeeze().to_sparse().to(cuda0)
                target = sample['target'].squeeze().to(cuda1)

                out = model(adjacency, atomtypes, memberships)

    #             loss = F.binary_cross_entropy(out, target, reduction='mean')

            #     print(torch.sum(target))

                loss = F.binary_cross_entropy(out, target)

            #     print(torch.sum(loss * target), torch.sum(target), torch.sum(loss * target) / torch.sum(target))

                loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * ((target - 1) + 2))  / torch.sum((target - 1) + 2)


                back_start = time.time()

                loss.backward()

                back_end = time.time()
                print('Backward: {}'.format(back_end - back_start))

                update_start = time.time()

                opt.step()

                update_end = time.time()
                print('Update: {}'.format(update_end - update_start))


                total_loss += loss.data

                pdb_id = sample['id'][0]
                out = out.data.to(cpu).numpy()
                target = target.to(cpu).numpy()

                if epoch % 1 == 0:
                    pool.apply_async(plot_channels, (pdb_id, target, out))

                del (adjacency, 
                     atomtypes, 
                     memberships, 
                     target,
                     out,
                     loss)

                torch.cuda.empty_cache()

                end = time.time()

                print('{}: {}'.format(sample['id'][0], end - start))


    #         if epoch % 100 == 0:    
            print('Epoch: {}, Loss: {:02f}'.format(epoch, float(total_loss) / 10))
                