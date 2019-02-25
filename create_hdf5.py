import numpy as np
import os
import h5py
import sys
import torch
from tqdm import *
import glob
from tesselate.data import make_sparse_mat

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

    # Convert to lowercase for consistency
    pdb_id = pdb_id.lower()
    
    # Get the paths to atomtypes and contacts files
    atomtype_file = os.path.join(dirname, pdb_id + '.atomtypes')
    contacts_file = os.path.join(dirname, pdb_id + '.contacts')
    
    # Make empty dicts for chains residues and atoms
    chains = {}
    residues = {}
    atoms = {}
    
    # Placeholders for the memberships and atom_ids arrays
    membership = []
    atom_ids = []

    with open(atomtype_file, 'r') as handle:
        
        # Loop through lines
        for line in handle.readlines():

            # Handle indices
            idx, atom = line.strip().split()
            idx = idx.split('/')
            chain_idx = idx[0]
            residue_idx = '/'.join(idx[:2])
            atom_idx = '/'.join(idx[:3])
            
            # Make sure chains are not duplicated
            if chain_idx not in chains:
                chains[chain_idx] = len(chains)

            # Make sure residues are not duplicated
            if residue_idx not in residues:
                residues[residue_idx] = len(residues)

            # Make sure unstructured atoms are not duplicated
            if atom_idx not in atoms:
                atoms[atom_idx] = len(atom_ids)

                # Generate the membership array and atom identity list
                membership.append((residues[residue_idx], atoms[atom_idx]))
                atom_ids.append((chains[chain_idx], residues[residue_idx], atoms[atom_idx], int(atom)))

    # Create empty list to hold contact info
    contacts = []

    # Open the contacts file
    with open(contacts_file, 'r') as handle:
        
        # Loop through lines
        for line in handle.readlines():
            
            # Split into partner 1, partner 2, contact
            split_line = line.strip().split()
            i = '/'.join(split_line[0].split('/')[:3])
            j = '/'.join(split_line[1].split('/')[:3])
            spectrum = [int(k) for k in split_line[2:]]
            for channel, val in enumerate(spectrum):
                if val == 1:
                    contacts.append([atoms[i], atoms[j], channel])
            
    return (np.array(atom_ids), np.array(membership), np.array(contacts))


# def make_sparse_mem_mat(idx_mat):
#     idx_mat = idx_mat.astype(np.int)
    
#     idx = torch.from_numpy(idx_mat)
#     val = torch.ones(len(idx))
    
#     out_mat = torch.sparse.FloatTensor(idx.t(), val, torch.Size([int(torch.max(idx[:, 0])) + 1, int(torch.max(idx[:, 1])) + 1]))
#     return(out_mat)


# def make_sparse_contact_mat(contacts_mat, atom_count):
#     contacts_mat = contacts_mat.astype(np.int)
#     idx = torch.from_numpy(contacts_mat)
#     val = torch.ones(len(idx))
#     out_mat = torch.sparse.FloatTensor(idx.t(), val, torch.Size([atom_count, atom_count, 15]))
#     return(out_mat)


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


def extract_sparse(sparse_mat):
    indices = sparse_mat._indices().t().numpy()
    values = sparse_mat._values().numpy()
    return(np.column_stack((indices, values)))
    

if __name__ == '__main__':
    
    h5file = h5py.File('./data/contacts.hdf5', 'a')
    
    if 'atomtypes' not in h5file:
        atomtypes_group = h5file.create_group('atomtypes')
        memberships_group = h5file.create_group('memberships')
        contacts_group = h5file.create_group('contacts')
        adjacency_group = h5file.create_group('adjacency')
        target_group = h5file.create_group('target')
        
    else:
        atomtypes_group = h5file['atomtypes']
        memberships_group = h5file['memberships']
        contacts_group = h5file['contacts']
        adjacency_group = h5file['adjacency']
        target_group = h5file['target']
    
    proc_target = sys.argv[1]
    
    if os.path.isdir(proc_target):
        atomtype_files = [i.replace('.atomtypes', '') for i in glob.glob('*.atomtypes')]
        contact_files = [i.replace('.contacts', '') for i in glob.glob('*.contacts')]
        accessions = [i for i in atomtype_files if i in contact_files]
        
        accessions = [acc.strip().lower() for acc in handle.readlines()]
        
    elif os.path.isfile(proc_target):
        with open(sys.argv[1], 'r') as handle:
            accessions = [acc.strip().lower() for acc in handle.readlines()]
            
    else:
        raise(ValueError('Invalid target: `{}` file or directory not found.'.format(proc_target)))
        
    for entry in tqdm(accessions):
        
        if entry not in atomtypes_group.keys():
            tqdm.write('Adding {}'.format(entry))
        
            atomtypes, memberships, contacts = process_arpeggio_out('./data', entry)
            
            atomtypes = atomtypes[:, -1].astype(np.uint8)
            memberships = memberships.astype(np.uint32)
            contacts = contacts.astype(np.uint32)

            atomtypes_group.create_dataset(entry, data=atomtypes, compression='gzip', compression_opts=9)
            memberships_group.create_dataset(entry, data=memberships, compression='gzip', compression_opts=9)
            contacts_group.create_dataset(entry, data=contacts, compression='gzip', compression_opts=9)

            # Get the 'memberships' and 'contacts' sparse matrices 
            memberships = make_sparse_mat(memberships, 'mem')
            contacts = make_sparse_mat(contacts, 'con', count=len(atomtypes))

            # Extract the covalent bonds between all atoms in the structure
            covalent = contacts.to_dense()[:, :, cont_chan['covalent']].squeeze()
            adjacency = create_adj(covalent).to_sparse()

            # Extract the contact maps for training
            contacts = contacts.to_dense()[:, :, cont_chan['vdw']:]
            memberships = memberships.to_dense()
            target = create_target(contacts, memberships, device).to_sparse()
            
            adjacency = extract_sparse(adjacency).astype(np.float16)
            target = extract_sparse(target)[:, :3].astype(np.uint32)

            adjacency_group.create_dataset(entry, data=adjacency, compression='gzip', compression_opts=9)
            target_group.create_dataset(entry, data=target, compression='gzip', compression_opts=9)
        
    h5file.close()
