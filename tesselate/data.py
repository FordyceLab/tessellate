import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset


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
            self.accessions = [acc.lower() for acc in handle.readlines()]
            
    def __len__(self):
        return len(self.accessions)
    
    def __getitem__(self, idx):
        
        h5file = h5py.File(self.hdf5_dataset, 'r')
        
        entry = self.accessions[idx]
        
        atomtypes = torch.from_numpy(h5file['atomtypes'][entry][:])
        memberships = h5file['memberships'][entry][:]
        contacts = h5file['contacts'][entry][:]
        
        memberships = make_sparse_mem_mat(memberships)
        contacts = make_sparse_contact_mat(contacts, len(atomtypes))
        
        return (atomtypes, memberships, contacts)
    
