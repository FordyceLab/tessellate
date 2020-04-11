from matplotlib import pyplot as plt
import numpy as np

def protein_size_from_target(target_len):
    return int(np.roots([0.5, 0.5, -target_len])[1])


def restructure_target(target):
    prot_size = protein_size_from_target(len(target))
    restructured_target = np.zeros((12, prot_size, prot_size))
    
    idx = 0
    for i, j in zip(*np.triu_indices(prot_size)):
        restructured_target[:, i, j] = target[idx]
        
        if i != j:
            restructured_target[:, j, i] = target[idx]
            
        idx += 1
        
    return restructured_target


def remap_and_plot(pdb_id, target, out, epoch):
    target = restructure_target(target)
    out = restructure_target(out)
    plot_channels(pdb_id, target, out, epoch)

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