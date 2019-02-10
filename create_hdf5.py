import numpy as np
import os
import h5py
import sys
import torch
from tqdm import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
    
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


def process_arpeggio_out(dirname, pdb_id):

    pdb_id = pdb_id.lower()
    
    atomtype_file = os.path.join(dirname, pdb_id + '.atomtypes')
    contacts_file = os.path.join(dirname, pdb_id + '.contacts')
    
    residues = {}
    atoms = {}
    membership = []
    atom_ids = []

    with open(atomtype_file, 'r') as handle:
        for line in handle.readlines():

            # Handle indices
            idx, atom = line.strip().split()
            idx = idx.split('/')
            residue_idx = '/'.join(idx[:2])
            atom_idx = '/'.join(idx[:3])

            # Make sure residues are not duplicated
            if residue_idx not in residues:
                residues[residue_idx] = len(residues)

            # Make sure unstructured atoms are not duplicated
            if atom_idx not in atoms:
                atoms[atom_idx] = len(atom_ids)

                # Generate the membership array and atom identity list
                membership.append((residues[residue_idx], atoms[atom_idx]))
                atom_ids.append(int(atom))

    contacts = []

    with open(contacts_file, 'r') as handle:    
        for line in handle.readlines():
            split_line = line.strip().split()
            i = '/'.join(split_line[0].split('/')[:3])
            j = '/'.join(split_line[1].split('/')[:3])
            spectrum = int(''.join(split_line[2:]), 2)
            contacts.append([atoms[i], atoms[j], spectrum])
            
    return (np.array(atom_ids), np.array(membership), np.array(contacts))


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


def create_adj(covalent):
    # Make the normalized adjacency matrix
    C_hat = covalent + torch.eye(covalent.shape[0])
    diag = 1 / torch.sqrt(C_hat.sum(dim=1))
    D_hat = torch.zeros_like(C_hat)
    n = D_hat.shape[0]
    D_hat[range(n), range(n)] = diag

    adjacency = D_hat.mm(C_hat).mm(D_hat)
    
    return adjacency


def create_target(contacts, memberships, device):
    # Transfer everything to the GPU for faster processing
    memberships = memberships.to(device)

    targets = []
    
    # Calculate the targets
    for i in range(contacts.shape[2]):
        contact = contacts[:, :, i].squeeze().to(device)
        
        targets.append((memberships
                        .mm(contact)
                        .mm(memberships.transpose(0, 1)) > 0)
                       .float().cpu())
    
    target = torch.stack(targets, dim=0)
    
    return target
    

if __name__ == '__main__':
    
    h5file = h5py.File('./data/contacts.hdf5', 'w')
    
    atomtypes_group = h5file.create_group('atomtypes')
    memberships_group = h5file.create_group('memberships')
    contacts_group = h5file.create_group('contacts')
    adjacency_group = h5file.create_group('adjacency')
    target_group = h5file.create_group('target')
    
    with open(sys.argv[1], 'r') as handle:
        accessions = [acc.strip().lower() for acc in handle.readlines()]
        
    for entry in tqdm(accessions):
        
        atomtypes, memberships, contacts = process_arpeggio_out('./data', entry)
        
        atomtypes_group.create_dataset(entry, data=atomtypes, compression='gzip', compression_opts=9)
        memberships_group.create_dataset(entry, data=memberships, compression='gzip', compression_opts=9)
        contacts_group.create_dataset(entry, data=contacts, compression='gzip', compression_opts=9)
        
        # Get the 'memberships' and 'contacts' dense matrices 
        memberships = make_sparse_mem_mat(memberships).to_dense()
        contacts = make_sparse_contact_mat(contacts, len(atomtypes)).to_dense()

        # Extract the covalent bonds between all atoms in the structure
        covalent = contacts[:, :, cont_chan['covalent']]
        adjacency = create_adj(covalent)
        
        # Extract the contact maps for training
        contacts = contacts[:, :, cont_chan['vdw']:]
        target = create_target(contacts, memberships, device)
        
        adjacency_group.create_dataset(entry, data=adjacency, compression='gzip', compression_opts=9)
        target_group.create_dataset(entry, data=target, compression='gzip', compression_opts=9)
        
    h5file.close()
