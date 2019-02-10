import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset

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

def make_sparse_mem_mat(idx_mat):
    idx = torch.from_numpy(idx_mat)
    val = torch.ones(len(idx))
    out_mat = torch.sparse.FloatTensor(idx.t(), val, torch.Size([torch.max(idx[:, 0]) + 1, torch.max(idx[:, 1]) + 1]))
    return(out_mat)


def make_sparse_contact_mat(contacts_mat, atom_count):
    full_contacts_mat = []
    for i in range(len(contacts_mat)):
        expanded_conts = [idx for idx, cont in enumerate(bin(contacts_mat[i, 2])[2:].zfill(15)) if cont == '1']
        full_contacts_mat.extend([[contacts_mat[i, 0], contacts_mat[i, 1], expanded_conts[j]] for j in range(len(expanded_conts))])
        full_contacts_mat.extend([[contacts_mat[i, 1], contacts_mat[i, 0], expanded_conts[j]] for j in range(len(expanded_conts))])
    
    full_contacts_mat = np.array(full_contacts_mat)
    
    idx = torch.from_numpy(full_contacts_mat)
    val = torch.ones(len(idx))
    out_mat = torch.sparse.FloatTensor(idx.t(), val, torch.Size([atom_count, atom_count, 15]))
    return(out_mat)


class TesselateDataset(Dataset):
    
    def __init__(self, accession_list, hdf5_dataset):
        
        self.accession_list = accession_list
        self.hdf5_dataset = hdf5_dataset
        
        with open(accession_list, 'r') as handle:
            self.accessions = [acc.strip().lower() for acc in handle.readlines()]
            
    def __len__(self):
        return len(self.accessions)
    
    def __getitem__(self, idx):
        
        # Get the entry PDB ID
        entry = self.accessions[idx]
        
        with h5py.File(self.hdf5_dataset, 'r') as h5file:

            # Read the data from the HDF5 file
            atomtypes = torch.from_numpy(h5file['atomtypes'][entry][:])
            memberships = h5file['memberships'][entry][:]
            adjacency = h5file['adjacency'][entry][:]
            target = h5file['target'][entry][:]

        # Return the data for training
        return {
            'id': entry,
            'atomtypes': atomtypes,
            'memberships': make_sparse_mem_mat(memberships).to_dense(),
            'adjacency': adjacency,
            'target': target
        }
    