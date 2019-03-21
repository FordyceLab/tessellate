import numpy as np
import os
import h5py
import sys
import torch
from tqdm import *
import glob
from tesselate.data import make_sparse_mat
from concurrent.futures import ThreadPoolExecutor
    
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
    
    # Create membership dictionary
    mem_dict = {}

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
    contacts = set()
    covalent = set()
    target = set()
    

    # Open the contacts file
    with open(contacts_file, 'r') as handle:
        
        # Loop through lines
        for line in handle.readlines():
            
            # Split into partner 1, partner 2, contact
            split_line = line.strip().split()
            
            atom_i = '/'.join(split_line[0].split('/')[:3])
            atom_j = '/'.join(split_line[1].split('/')[:3])
            
            res_i = '/'.join(split_line[0].split('/')[:2])
            res_j = '/'.join(split_line[1].split('/')[:2])
            
            spectrum = [int(k) for k in split_line[2:]]
            
            atom_partner_1 = atoms[atom_i]
            atom_partner_2 = atoms[atom_j]
            
            res_partner_1 = residues[res_i]
            res_partner_2 = residues[res_j]
            
            for channel, val in enumerate(spectrum):
                if val == 1:
                    contacts.add((atom_partner_1, atom_partner_2, channel))
                    contacts.add((atom_partner_2, atom_partner_1, channel))
                    
                    if channel == cont_chan['covalent']:
                        covalent.add((atom_partner_1, atom_partner_2))
                        covalent.add((atom_partner_2, atom_partner_1))
                        
                    if channel >= cont_chan['vdw']:
                        target.add((res_partner_1, res_partner_2, channel))
                        target.add((res_partner_1, res_partner_2, channel))
                        
    return (np.array(atom_ids), np.array(membership), np.array(list(contacts)),
            np.array(list(target)), np.array(list(covalent)))


def create_adj(covalent):
    # Make the normalized adjacency matrix
    C_hat = covalent + torch.eye(covalent.shape[0])
    diag = 1 / torch.sqrt(C_hat.sum(dim=1))
    D_hat = torch.zeros_like(C_hat)
    n = D_hat.shape[0]
    D_hat[range(n), range(n)] = diag
    
    D_hat_sparse = D_hat.to_sparse()

    adjacency = D_hat_sparse.mm(C_hat).to_sparse().mm(D_hat).to_sparse()
    
    return adjacency


def extract_sparse(sparse_mat):
    indices = sparse_mat._indices().t().numpy()
    values = sparse_mat._values().numpy()
    return(np.column_stack((indices, values)))


def process(entry):        
    atomtypes, memberships, contacts, target, covalent = process_arpeggio_out('data/processed', entry)
    
    if len(covalent.shape) == 2:

        atomtypes = atomtypes.astype(np.uint32)
        memberships = memberships.astype(np.uint32)
        contacts = contacts.astype(np.uint32)
        target = target.astype(np.uint32)

        # Get the 'memberships' and 'covalent' sparse matrices
        covalent = make_sparse_mat(covalent, 'cov', count=len(atomtypes)).to_dense()

        # Extract the covalent bonds between all atoms in the structure
        adjacency = create_adj(covalent)


        adjacency = extract_sparse(adjacency).astype(np.float32)

        return {
            'entry': entry,
            'atomtypes': atomtypes,
            'memberships': memberships,
            'contacts': contacts,
            'target': target,
            'adjacency': adjacency
        }

def hdf5_write(result, h5file):
#     tqdm.write('Adding {}...'.format(result['entry']))
    entry_group = h5file.create_group(result['entry'])
    entry_group.create_dataset('atomtypes', data=result['atomtypes'], compression='gzip', compression_opts=9)
    entry_group.create_dataset('memberships', data=result['memberships'], compression='gzip', compression_opts=9)
    entry_group.create_dataset('contacts', data=result['contacts'], compression='gzip', compression_opts=9)
    entry_group.create_dataset('target', data=result['target'], compression='gzip', compression_opts=9)
    entry_group.create_dataset('adjacency', data=result['adjacency'], compression='gzip', compression_opts=9)


if __name__ == '__main__':
    
    # Init the pool
    pool = ThreadPoolExecutor(80)
    
    # Open up the HDF5 file
    h5file = h5py.File('data/contacts.hdf5', 'a')
    
    # Read in the target list to be processed
    proc_target = sys.argv[1]
    
    # Get all possible targets from directory if proc_target is directory
    if os.path.isdir(proc_target):
        atomtype_files = [os.path.basename(i).replace('.atomtypes', '') for i in glob.glob(os.path.join(proc_target, '*.atomtypes'))]
        contact_files = [os.path.basename(i).replace('.contacts', '') for i in glob.glob(os.path.join(proc_target, '*.contacts'))]
        accessions = [i for i in atomtype_files if i in contact_files]
        
    # Read in the text list of targets if proc_target is file
    elif os.path.isfile(proc_target):
        with open(proc_target, 'r') as handle:
            accessions = [acc.strip().lower() for acc in handle.readlines()]
            
    else:
        raise(ValueError('Invalid target: `{}` file or directory not found.'.format(proc_target)))
    
    futures = []
    
    # Loop through the accession numbers
    for entry in tqdm(accessions, leave=False):
        if entry not in h5file:
            futures.append(pool.submit(process, (entry)))
        
    initial_len = len(futures)
    
    while len(futures) > 0:
        tqdm.write('{} of {} structures remaining.'.format(len(futures), initial_len))
        for idx, future in tqdm(enumerate(futures), total=len(futures), leave=False):
            if future.done():
                if future.result() is not None:
                    hdf5_write(future.result(), h5file)
                del futures[idx]
    
    h5file.close()
