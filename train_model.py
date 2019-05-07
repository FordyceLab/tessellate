from git import Repo
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tesselate.data import TesselateDataset, make_sparse_mat
from tesselate.model import Network
from tesselate.plot import remap_and_plot
import torch
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import *
import time
import wandb
    
    
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
    
    monitor = True
    
    # Check to make sure the repo is clean
    # Since we are logging git commits to track model changes over time
    if monitor:
        repo = Repo('.')
        if repo.is_dirty():
            print("Git repo is dirty, please commit changes before training model.")
            sys.exit(1)
    
    # Initialize the multiprocessing capabilities for plotting
#     multiprocessing.set_start_method('spawn')
#     queue = mp.Queue()
#     p = mp.Pool(10, plot_worker, (queue,))

    WANDB = monitor
    
    if WANDB:
        wandb.init(project='tesselate', config={'commit': repo.head.object.hexsha})
    
    
    # Define the model parameters
    INPUT_SIZE = 10
    GRAPH_CONV = 5
    
    # Get references to the different devices
    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')
    cpu = torch.device('cpu')

    # Genetrate the model
    model = Network(INPUT_SIZE, GRAPH_CONV, GRAPH_CONV, cuda0, cuda1)
    
    if WANDB:
        wandb.watch(model, log='all')

    # Generate the dataset/dataloader for training
#     data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/success/training_30_ids.txt',
#                             'data/training.hdf5')
    data = TesselateDataset('test2.txt',
                            'data/training.hdf5')
    dataloader = DataLoader(data, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False,
                            collate_fn=dict_collate)

#     val_data = TesselateDataset('id_lists/ProteinNet/ProteinNet12/x_ray/success/validation_ids.txt',
#                                 'data/training.hdf5')
    val_data = data
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=False,
                            collate_fn=dict_collate)
    

    # Initialize the optimizer
#     opt = optim.SGD(model.parameters(), lr = .05, momentum=0.9) #, weight_decay=1e-4)
    opt = optim.Adam(model.parameters())

    step_iter = 0
    step_loss = 0
    step_count = 0
    
    # Train for N epochs
    
    OOM_COUNT=0
    for epoch in trange(100 * 1000, leave=False):
        
        model.train()
        
        # Sum the total loss
        total_loss = 0
        total_count = 0

        # For each sample in each example, train the model
        for sample in tqdm(dataloader, leave=False, dynamic_ncols=True):

                # Zero the gradient
                opt.zero_grad()
                
                atomtypes = sample['atomtypes']
                atom_adjacency = sample['atom_adjacency']
                memberships = sample['memberships']
                res_adjacency = sample['res_adjacency']
                target = sample['target']
                combos = sample['combos']

                # Move the data to the appropriate device
                atom_adjacency = atom_adjacency.float().to(cuda0)
                memberships = memberships.float().to(cuda0)
                res_adjacency = res_adjacency.float().to(cuda0)
                target = target.float().to(cuda0)
                combos = combos.float().to(cuda0)

                try:
                    # Make the prediction
                    out = model(atom_adjacency, res_adjacency, atomtypes, memberships, combos)

                    # Get the frequency-adjusted loss
                    loss = F.binary_cross_entropy(out, target, reduction='none')
                    loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * torch.abs(target - 1))  / torch.sum(torch.abs(target - 1))

                    # Make the backward pass
                    loss.backward()

                    # Step the optimizer
                    opt.step()

                    # Get the total loss
                    total_loss += loss.data
                    total_count += 1

                    step_loss += loss.data
                    step_count += 1
                    step_iter += 1

                    if step_iter % 1000 == 0 and WANDB:
                        step_loss = step_loss / step_count
                        wandb.log({'step_loss': step_loss})

                        step_loss = 0
                        step_count = 0

                        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'step_{}.pt'.format(step_iter)))
                    
                except RuntimeError:
                    OOM_COUNT += 1
                    tqdm.write('OOM: {}'.format(OOM_COUNT))

        train_loss = total_loss / total_count
        
        if WANDB:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'epoch_{}.pt'.format(epoch)))
            
        total_count = 0
        total_loss = 0
        
        if epoch % 100 != 0:
            if WANDB:
                wandb.log({'train_loss': train_loss})
            
        else:
            model.eval()

            pred_histograms = {}

            for sample in tqdm(val_loader, leave=False, dynamic_ncols=True):

                atomtypes = sample['atomtypes']
                atom_adjacency = sample['atom_adjacency']
                memberships = sample['memberships']
                res_adjacency = sample['res_adjacency']
                target = sample['target']
                combos = sample['combos']

                # Move the data to the appropriate device
                atom_adjacency = atom_adjacency.float().to(cuda0)
                memberships = memberships.float().to(cuda0)
                res_adjacency = res_adjacency.float().to(cuda0)
                target = target.float().to(cuda0)
                combos = combos.float().to(cuda0)

                try:
                    # Make the prediction
                    out = model(atom_adjacency, res_adjacency, atomtypes, memberships, combos)



                    # Get the frequency-adjusted loss
                    loss = F.binary_cross_entropy(out, target, reduction='none')
                    loss = torch.sum(loss * target) / torch.sum(target) + torch.sum(loss * torch.abs(target - 1))  / torch.sum(torch.abs(target - 1))

                    # Get the total loss
                    total_loss += loss.data
                    total_count += 1

                    # Extract data for plotting
                    pdb_id = sample['id']
                    out = out.data.to(cpu).numpy()
                    target = target.to(cpu).numpy()
                    remap_and_plot(pdb_id, target, out, epoch)

                    if WANDB:
                        for i in range(12):
                            pred_histograms['{}_pred_channel_{}'.format(pdb_id, i)] = wandb.Histogram(out[:, i])

                except RuntimeError:
                    continue

            val_loss = total_loss / total_count

            if WANDB:
                log_dict = {'train_loss': train_loss, 'val_loss': val_loss}

                if pred_histograms:
                    log_dict.update(pred_histograms)

                wandb.log(log_dict)

    
#     # Finish the plotting queue
#     queue.put('plot_complete')
#     queue.close()
#     queue.join_thread()
#     p.join()
                