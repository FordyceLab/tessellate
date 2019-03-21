import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset
from itertools import combinations

cont_chan = (
    'clash',
    'covalent',
    'vdw_clash',
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

cont_chan = {btype: chan_num for chan_num, btype in enumerate(cont_chan)}


def torch_idx(idx_mat):
    idx_mat = idx_mat.astype(np.int)
    idx = torch.from_numpy(idx_mat)
    return idx


def make_sparse_mat(idx_mat, mat_type, count=None):
    """
    Args:
    - id_mat (ndarray) - Numpy array containing the sparse matrix.
        All coordinates assumed to have value = 1 unless type is 'adj'
    - mat_type (str) - string indicating matrix type, one of ['mem', 'con', 'adj', 'tar', 'cov']
    """
    
    if mat_type not in ['mem', 'con', 'adj', 'tar', 'cov']:
        raise(ValueError('`mat_type` = {}, unrecognized mat_type.'.format(mat_type)))
    
    if mat_type == 'adj':
        idx = torch_idx(idx_mat[:, :2])
        val = torch.from_numpy(idx_mat[:, 2].astype(np.float))
        size = torch.Size([int(torch.max(idx)) + 1, int(torch.max(idx)) + 1])
    else:
        idx = torch_idx(idx_mat)
        val = torch.ones(len(idx))
        
        if mat_type =='mem':
            size = torch.Size([int(torch.max(idx[:, 0])) + 1, int(torch.max(idx[:, 1])) + 1])
        elif mat_type == 'con':
            size = torch.Size([count, count, 15])
        elif mat_type == 'tar':
            size = torch.Size([12, count, count])
        elif mat_type == 'cov':
            size = torch.Size([count, count])
        else:
            raise()
        
    out_mat = torch.sparse.FloatTensor(idx.t(), val, size)
    return(out_mat)


def select_chains(atomtypes, memberships, adjacency, target, chains):
    
    CHAIN_IDX = 0
    RESIDUE_IDX = 1
    ATOM_IDX = 2
    
    sel_atomtypes = np.array([row for row in atomtypes if row[0] in chains])
    
    sel_residues = np.unique(sel_atomtypes[:, RESIDUE_IDX])
    sel_atoms = np.unique(sel_atomtypes[:, ATOM_IDX])
    target_channels = range(target.shape[0])
    
    sel_adjacency = adjacency[np.ix_(sel_atoms, sel_atoms)]
    sel_memberships = memberships[np.ix_(sel_residues, sel_atoms)]
    sel_target = target[np.ix_(target_channels, sel_residues, sel_residues)]
    
    return (sel_atomtypes, sel_memberships, sel_adjacency, sel_target)


class TesselateDataset(Dataset):
    """
    feed_method - one of 'single_chains', 'pairwise', 'complete'
    """
    
    def __init__(self, accession_list, hdf5_dataset, feed_method='complete'):
        
        self.accession_list = accession_list
        self.hdf5_dataset = hdf5_dataset
        self.feed_method = feed_method
        self.in_memory = {}
        
        with open(accession_list, 'r') as handle:
            self.accessions = [acc.strip().lower() for acc in handle.readlines()]
            
        for entry in self.accessions:
        
            with h5py.File(self.hdf5_dataset, 'r') as h5file:

                # Read the data from the HDF5 file
                atomtypes = h5file['atomtypes'][entry][:]
                memberships = h5file['memberships'][entry][:]
                adjacency = h5file['adjacency'][entry][:]
                target = h5file['target'][entry][:]

            memberships = make_sparse_mem_mat(memberships).to_dense().numpy()

            unique_chains = np.unique(atomtypes[:, 0])

            chain_combos = []

            entry_list = []
            atomtype_list = []
            membership_list = []
            adjacency_list = []
            target_list = []

            if self.feed_method == 'complete':
                entry_list.append(entry)
                atomtype_list.append(atomtypes)
                membership_list.append(memberships)
                adjacency_list.append(adjacency)
                target_list.append(target)

            else:

                if self.feed_method == 'single_chain':
                    for chain in unique_chains:
                        chain_combos.append([chain])

                elif self.feed_method == 'pairwise':
                    chain_combos.extend(combinations(unique_chains, 2))

                else:
                    raise(ValueError('Unknown feed method "{}".'.format(self.feed_method)))

                for chains in chain_combos:
                    selection = select_chains(atomtypes, memberships, adjacency, target, chains)

                    entry_list.append('{}_{}'.format(entry, '_'.join([str(i) for i in chains])))
                    atomtype_list.append(selection[0])
                    membership_list.append(selection[1])
                    adjacency_list.append(selection[2])
                    target_list.append(selection[3])
                    
            data_dict = {
                'id': entry_list,
                'atomtypes': [torch.from_numpy(i) for i in atomtype_list],
                'memberships': [torch.from_numpy(i) for i in membership_list],
                'adjacency': [torch.from_numpy(i) for i in adjacency_list],
                'target': [torch.from_numpy(i) for i in target_list]
            }
            
            self.in_memory[entry] = data_dict
            
    def __len__(self):
        return len(self.accessions)
    
    def __getitem__(self, idx):
        
        # Get the entry PDB ID
        entry = self.accessions[idx]
        if entry in self.in_memory:
            return self.in_memory[entry]
        
        ###################
#         else:
        
        with h5py.File(self.hdf5_dataset, 'r') as h5file:

            # Read the data from the HDF5 file
            atomtypes = h5file['atomtypes'][entry][:]
            memberships = h5file['memberships'][entry][:]
            adjacency = h5file['adjacency'][entry][:]
            target = h5file['target'][entry][:]

        memberships = make_sparse_mem_mat(memberships).to_dense().numpy()

        unique_chains = np.unique(atomtypes[:, 0])

        chain_combos = []

        entry_list = []
        atomtype_list = []
        membership_list = []
        adjacency_list = []
        target_list = []

        if self.feed_method == 'complete':
            entry_list.append(entry)
            atomtype_list.append(atomtypes)
            membership_list.append(memberships)
            adjacency_list.append(adjacency)
            target_list.append(target)

        else:

            if self.feed_method == 'single_chain':
                for chain in unique_chains:
                    chain_combos.append([chain])

            elif self.feed_method == 'pairwise':
                chain_combos.extend(combinations(unique_chains, 2))

            else:
                raise(ValueError('Unknown feed method "{}".'.format(self.feed_method)))

            for chains in chain_combos:
                selection = select_chains(atomtypes, memberships, adjacency, target, chains)

                entry_list.append('{}_'.format(entry, '_'.join([str(i) for i in chains])))
                atomtype_list.append(selection[0])
                membership_list.append(selection[1])
                adjacency_list.append(selection[2])
                target_list.append(selection[3])

        data_dict = {
            'id': entry_list,
            'atomtypes': [torch.from_numpy(i) for i in atomtype_list],
            'memberships': [torch.from_numpy(i) for i in membership_list],
            'adjacency': [torch.from_numpy(i) for i in adjacency_list],
            'target': [torch.from_numpy(i) for i in target_list]
        }

        self.in_memory[entry] = data_dict

        # Return the data for training
        return data_dict
    #########################