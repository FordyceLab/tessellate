from git import Repo
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tesselate.data import TesselateDataset, make_sparse_mat
from tesselate.model import Network
import torch
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import *
import time
import wandb

#multiprocessing.set_sharing_strategy('file_system')
# multiprocessing.set_sharing_strategy('file_descriptor')


def plot_channels(pdb_id, target, out, epoch):
    """
    Subroutine for plotting all of the channels in a contact map.
    
    Args:
    - pdb_id (str) - ID of the target structure.
    - target (numpy ndarray) - Array containing the prediction target.
    - out (numpy ndarray) - Array containing the model output.
    - epoch (int) - Epoch number for the model.
    """
    
    # Make contact channel dictionary
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
    
    # Generate the figure and axis references
    fig, ax = plt.subplots(target.shape[0],
                           3,
                           figsize=(2 * 3, 2 * target.shape[0]))
    
    # Add a title
    fig.suptitle('{} - Epoch: {:06d}'.format(pdb_id, epoch), fontsize=16)

    # Loop through each channel
    for i in range(0, target.shape[0] * 3, 3):
        
        # Get the column ID
        j = int(i / 3)

        # Make the pred, thres_pred, target plots
        ax1 = plt.subplot(target.shape[0], 3, i + 1)
        ax1.imshow(out[j], cmap='gray', vmin=0, vmax=1)
        
        ax2 = plt.subplot(target.shape[0], 3, i + 2)
        ax2.imshow(out[j].round(0), cmap='gray', vmin=0, vmax=1)
        ax2.annotate(cont_chan[j], xy=(0.5, 1), xytext=(0, 10),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
        
        ax3 = plt.subplot(target.shape[0], 3, i + 3)
        ax3.imshow(target[j], cmap='gray', vmin=0, vmax=1)
        
    # Fix the plot layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot to the figures directory
    plt.savefig('figs/{}_{:06d}.png'.format(pdb_id, epoch))
    plt.close()
    
    
def plot_worker(queue):
    """
    Helper function to draw plotting examples from a queue and plot them.
    
    Args:
    - 
    """
    # Draw an example from the queue
    example = queue.get()
    
    # Keep drawing until queue is complete
    while example != 'plot_complete':
        
        # Make sure that the example contains 4 datasets
        if len(example) == 4:
            
            # Make the plots in parallel
            plot_channels(*example)
        
        # Get the next example
        example = queue.get()
    
def dict_collate(batch):
    return batch[0]

if __name__ == '__main__':
    
    # Check to make sure the repo is clean
    # Since we are logging git commits to track model changes over time
    repo = Repo('.')
    if repo.is_dirty():
        print("Git repo is dirty, please commit changes before training model.")
        sys.exit(1)
    
    # Initialize the multiprocessing capabilities for plotting
#     multiprocessing.set_start_method('spawn')
#     queue = mp.Queue()
#     p = mp.Pool(10, plot_worker, (queue,))

    WANDB = True
    
    if WANDB:
        wandb.init(project='tesselate', config={'commit': repo.head.object.hexsha})
    
    
    # Define the model parameters
    INPUT_SIZE = 10
    GRAPH_CONV = 15
    FEED = 'complete'
    
    # Get references to the different devices
    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')
    cpu = torch.device('cpu')

    # Genetrate the model
    model = Network(INPUT_SIZE, GRAPH_CONV, cuda0, cuda1)
    
    if WANDB:
        wandb.watch(model)

    # Generate the dataset/dataloader for training
    data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/success/training_30_ids.txt', 'data/contacts.hdf5', FEED)
    dataloader = DataLoader(data, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False,
                            collate_fn=dict_collate)

    val_data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/success/validation_ids.txt', 'data/contacts.hdf5', FEED)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False,
                            collate_fn=dict_collate)
    

    # Initialize the optimizer
    opt = optim.SGD(model.parameters(), lr = .01, momentum=0.9) #, weight_decay=1e-4)

    step_iter = 0
    step_loss = 0
    step_count = 0
    
    # Train for N epochs
    for epoch in trange(1000, leave=False):
        
        # Sum the total loss
        total_loss = 0
        total_count = 0

        # For each sample in each example, train the model
        for sample in tqdm(dataloader, leave=False):
            for idx in tqdm(range(len(sample['id'])), leave=False):

                # Zero the gradient
                opt.zero_grad()
                
                # Extract the data and convert to appropriate tensor format
                atomtypes = torch.from_numpy(sample['atomtypes'][idx][:, 3])
                adjacency = make_sparse_mat(sample['adjacency'][idx], 'adj')
                memberships = make_sparse_mat(sample['memberships'][idx], 'mem')
                target = torch.from_numpy(sample['target'][idx])
                
                combos = torch.from_numpy(sample['combos'][idx])
                
                combos = torch.sparse.FloatTensor(combos.t(),
                                                  torch.ones(len(combos)) * 0.5,
                                                  torch.Size((len(combos), len(memberships))))                

                # Move the data to the appropriate device
                adjacency = adjacency.float().to(cuda0)
                memberships = memberships.float().to(cuda0)
                target = target.float().to(cuda1)
                combos = combos.float().to(cuda0)

                try:
                    # Make the prediction
                    out = model(adjacency, atomtypes, memberships, combos)

                    # Get the mean reduced loss
    #                     loss = F.binary_cross_entropy(out, target, reduction='mean')

                    # Get the frequency-adjusted loss
                    loss = F.binary_cross_entropy(out, target, reduction='none')
                    loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * ((target - 1) + 2))  / torch.sum((target - 1) + 2)

                    # Make the backward pass
                    loss.backward()

                    # Step the optimizer
                    opt.step()

                    # Get the summed loss
    #                     loss = F.binary_cross_entropy(out, target, reduction='sum')


                    # Get the total loss
                    total_loss += loss.data
                    total_count += target.shape[0] * target.shape[1]

                    step_loss += loss.data
                    step_count += target.shape[0] * target.shape[1]
                    step_iter += 1

                    if step_iter % 1000 == 0:
                        step_loss = step_loss / step_count
                        wandb.log({'step_loss': step_loss})

                        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'step_{}.pt'.format(step)))

                        step_loss = 0
                        step_count = 0
                    
                except:
                    continue
                    

        train_loss = total_loss / total_count
        
        if WANDB:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'epoch_{}.pt'.format(epoch)))
            
        total_count = 0
        total_loss = 0
        
        for sample in tqdm(val_loader):
            
            for idx in tqdm(range(len(sample['id'])), leave=False):

                # Extract the data and convert to appropriate tensor format
                atomtypes = torch.from_numpy(sample['atomtypes'][idx][:, 3])
                adjacency = make_sparse_mat(sample['adjacency'][idx], 'adj')
                memberships = make_sparse_mat(sample['memberships'][idx], 'mem')
                target = torch.from_numpy(sample['target'][idx])
                
                combos = torch.from_numpy(sample['combos'][idx])
                
                combos = torch.sparse.FloatTensor(combos.t(),
                                                  torch.ones(len(combos)) * 0.5,
                                                  torch.Size((len(combos), len(memberships))))
                
                # Move the data to the appropriate device
                adjacency = adjacency.float().to(cuda0)
                memberships = memberships.float().to(cuda0)
                target = target.float().to(cuda1)
                combos = combos.float().to(cuda0)
                
                try:
                    # Make the prediction
                    out = model(adjacency, atomtypes, memberships, combos)

                    # Get the summed loss
#                     loss = F.binary_cross_entropy(out, target, reduction='sum')
                    loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * torch.abs(target - 1))  / torch.sum(torch.abs(target - 1))

                    # Get the total loss
                    total_loss += loss.data
                    total_count += target.shape[0] * target.shape[1]

                    # Extract data for plotting
#                     pdb_id = sample['id'][idx]
#                     out = out.data.to(cpu).numpy()
#                     target = target.to(cpu).numpy()
                    
#                     queue.put((pdb_id, target, out, epoch))
                
                except:
                    continue
                
        val_loss = total_loss / total_count
        
        if WANDB:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

    
#     # Finish the plotting queue
#     queue.put('plot_complete')
#     queue.close()
#     queue.join_thread()
#     p.join()
                