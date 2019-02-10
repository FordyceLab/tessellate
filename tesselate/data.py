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
            contacts = h5file['contacts'][entry][:]

        # Get the 'memberships' and 'contacts' dense matrices 
        memberships = make_sparse_mem_mat(memberships).to_dense()
        contacts = make_sparse_contact_mat(contacts, len(atomtypes)).to_dense()

        # Extract the covalent bonds between all atoms in the structure
        covalent = contacts[:, :, 1]

        # Make the normalized adjacency matrix
        C_hat = covalent + torch.eye(covalent.shape[0])
        diag = 1 / torch.sqrt(C_hat.sum(dim=1))
        D_hat = torch.zeros_like(C_hat)
        n = D_hat.shape[0]
        D_hat[range(n), range(n)] = diag

        adjacency = D_hat.mm(C_hat).mm(D_hat)

        # Extract the contact maps for training
        contacts = contacts[:, :, 3:]

        # Tile the membership matrix for multiplication with each channel
        dense_mem = memberships.repeat(contacts.shape[2], 1, 1)

        # Transfer everything to the GPU for faster processing
        device = torch.device('cuda:1')
        dense_mem = dense_mem.to(device)
        contacts = contacts.to(device)

        # Calculate the targets
        target = (dense_mem
                  .bmm(contacts.transpose(0,2))
                  .bmm(dense_mem.transpose(1, 2)) > 0).float().cpu()

        # Return the data for training
        return {
            'id': entry,
            'atomtypes': atomtypes,
            'memberships': memberships,
            'adjacency': adjacency,
            'target': target
        }
    