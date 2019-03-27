import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset
from itertools import combinations

# Make a disctionary of the contact map channels
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
    """
    Extract the coordinates from a sparse matrix format and return as a torch.LongTensor.
    """
    idx_mat = idx_mat.astype(np.int)
    idx = torch.from_numpy(idx_mat)
    return idx


def extract_sparse(sparse_mat):
    """
    Extract a coordinate levele representation as a tuple for a PyTorch
    sparse tensor.
    
    Args:
    - sparse_mat (PyTorch sparse tensor) - The sparse matrix from which to
        extract coordinates and values.
        
    Returns:
    - A tuple of the indices (coordinates) of non-zero values and a
        vector of the corresponding values.
    """
    # Perform extraction
    indices = sparse_mat._indices().t().numpy()
    values = sparse_mat._values().numpy()
    
    # Return tuple
    return (indices, values)


def make_sparse_mat(idx_mat, mat_type, count=None):
    """
    Args:
    - id_mat (ndarray) - Numpy array containing the sparse matrix.
        All coordinates assumed to have value = 1 unless type is 'adj'
    - mat_type (str) - String indicating matrix type, one of ['mem', 'con', 'adj', 'tar', 'cov']
    - count (int) - Dimension of matrix for target and covalent ('tar' and 'cov')
        matrices only
        
    Returns:
    - A PyTorch sparse tensor representation of the desired tensor.
    """
    
    # Raise an error if the matrix is of an unrecognized type
    if mat_type not in ['mem', 'con', 'adj', 'tar', 'cov']:
        raise(ValueError('`mat_type` = {}, unrecognized mat_type.'.format(mat_type)))
    
    # Handle an adjacency matrix (values not all == 1)
    if mat_type == 'adj':
        # Get the indices and convert to int
        idx = torch_idx(idx_mat[:, :2])
        
        # Get the values and convert to float
        val = torch.from_numpy(idx_mat[:, 2].astype(np.float))
        
        # Get the size of the tensor (matrix)
        size = torch.Size([int(torch.max(idx)) + 1, int(torch.max(idx)) + 1])
        
    # Handle all other tensor types
    else:
        # Get the indices
        idx = torch_idx(idx_mat)
        
        # All values are 1
        val = torch.ones(len(idx))
        
        # Handle membership matrix
        # (size: RES_COUNT x ATOM_COUNT)
        if mat_type =='mem':
            size = torch.Size([int(torch.max(idx[:, 0])) + 1, int(torch.max(idx[:, 1])) + 1])
            
        # Handle contact matrix
        # (size: ATOM_COUNT x ATOM_COUNT)
        elif mat_type == 'con':
            size = torch.Size([count, count, 15])
            
        # Handle target matrix
        # (size: 12 x RES_COUNT x RES_COUNT)
        elif mat_type == 'tar':
            size = torch.Size([12, count, count])
            
        # Handle covalent matrix (target with only 1 channel for covalent)
        # (size: 12 x RES_COUNT x RES_COUNT)
        elif mat_type == 'cov':
            size = torch.Size([count, count])
       
    # Construct and return the sparse tensor 
    out_mat = torch.sparse.FloatTensor(idx.t(), val, size)
    return(out_mat)


def select_chains(atomtypes, memberships, adjacency, target, chains):
    """
    Select a subset of chains from the dataset encoding the entire structure.
    
    Args:
    - atomtypes (numpy ndarray) - ndarray representing the atomtypes
    - memberships (numpy ndarray) - ndarray representing the memberships
    - adjacency (numpy ndarray) - ndarray representing the adjacency matrix
    - target (numpy ndarray) - Dense ndarray representing the target
    - chains (list) - List of chains in selection
    
    Returns:
    - A tuple of dense ndarrays for the selection (atomtypes, memberships, adjacency, target)
    """
    
    # Hard code the column positions of the chain, residue, and atom identifiers in the atomtypes dataset.
    CHAIN_IDX = 0
    RESIDUE_IDX = 1
    ATOM_IDX = 2
    
    # Select the atomtypes
    sel_atomtypes = np.array([row for row in atomtypes if row[CHAIN_IDX] in chains])
    
    # Select the unique chains, residues, and atoms
    sel_chains = np.unique(sel_atomtypes[:, CHAIN_IDX])
    sel_residues = np.unique(sel_atomtypes[:, RESIDUE_IDX])
    sel_atoms = np.unique(sel_atomtypes[:, ATOM_IDX])
    
    # Select the pertinent information from the coordinate matrices
    #### NOTE: adjacency matrix may not be correct, may have to recompute for the smaller number of atoms when performing chain selection
    sel_adjacency = adjacency[[adjacency[i, 0] in sel_atoms and adjacency[i, 1] in sel_atoms for i in range(len(adjacency))]]
    sel_memberships = memberships[[memberships[i, 0] in sel_residues and memberships[i, 1] in sel_atoms for i in range(len(memberships))]]
    sel_target = target[[target[i, 1] in sel_residues and target[i, 2] in sel_residues for i in range(len(target))]]
    
    # Handle renumbering of atoms and residues
    chain_remap = {orig_number: idx for idx, orig_number in enumerate(np.sort(sel_chains))}
    res_remap = {orig_number: idx for idx, orig_number in enumerate(np.sort(sel_residues))}
    atom_remap = {orig_number: idx for idx, orig_number in enumerate(np.sort(sel_atoms))}
    
    # Perform all remappings
    sel_atomtypes[:, CHAIN_IDX] = [chain_remap[i] for i in sel_atomtypes[:, CHAIN_IDX]]
    sel_atomtypes[:, RESIDUE_IDX] = [res_remap[i] for i in sel_atomtypes[:, RESIDUE_IDX]]
    sel_atomtypes[:, ATOM_IDX] = [atom_remap[i] for i in sel_atomtypes[:, ATOM_IDX]]
    
    sel_adjacency[:, 0] = [atom_remap[i] for i in sel_adjacency[:, 0]]
    sel_adjacency[:, 1] = [atom_remap[i] for i in sel_adjacency[:, 1]]
    
    sel_memberships[:, 0] = [res_remap[i] for i in sel_memberships[:, 0]]
    sel_memberships[:, 1] = [atom_remap[i] for i in sel_memberships[:, 1]]
    
    sel_target[:, 1] = [res_remap[i] for i in sel_target[:, 1]]
    sel_target[:, 2] = [res_remap[i] for i in sel_target[:, 2]]
    
    # Return a tuple of the selections in coordinate form
    return (sel_atomtypes, sel_memberships, sel_adjacency, sel_target)


class TesselateDataset(Dataset):
    """
    Dataset class for structural data.
    
    Args:
    - accession_list (str) - File path from which to read PDB IDs for dataset.
    - hdf5_dataset (str) - File path of the HDF5 file containing the structural
        information.
    - feed_method (str) - One of 'single_chains', 'pairwise', or 'complete'
        indicating how to feed in the independent chains of the structures
        (default = 'complete').
    """
    
    def __init__(self, accession_list, hdf5_dataset, feed_method='complete'):
        
        # Store reference to accession list file
        self.accession_list = accession_list
        
        # Store reference to HDF5 data file
        self.hdf5_dataset = hdf5_dataset
        
        # Store desired feed method
        self.feed_method = feed_method
        
        # Read in and store a list of accession IDs
        with open(accession_list, 'r') as handle:
            self.accessions = [acc.strip().lower() for acc in handle.readlines()]
            
    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
        - Integer count of number of examples.
        """
        return len(self.accessions)
    
    def __getitem__(self, idx):
        """
        Get an item with a particular index value.
        
        Args:
        - idx (int) - Index of desired sample.
        
        Returns:
        - Dictionary of PDB ID and atomtype, memberships, adjacency, and
            target tensors. All tensors are sparse when possible.
        """
        
        # Get the entry PDB ID
        entry = self.accessions[idx]
        
        # Open the HDF5 data file
        with h5py.File(self.hdf5_dataset, 'r') as h5file:

            # Read the data from the HDF5 file
            atomtypes = h5file[entry]['atomtypes'][:].astype(np.int64)
            memberships = h5file[entry]['memberships'][:]
            adjacency = h5file[entry]['adjacency'][:]
            
            # Handle the target slightly differently
            # Rearrange columns
            target = h5file[entry]['target'][:][:, [2, 0, 1]]
            
            # Subtract 3 (because first 3 channels are dropped in the target)
            target[:, 0] = target[:, 0] - 3 

        # Extract the unique chains in the structure
        unique_chains = np.unique(atomtypes[:, 0])

        # Generate lists to store data combinations
        chain_combos = []
        
        entry_list = []
        atomtype_list = []
        membership_list = []
        adjacency_list = []
        target_list = []

        # Handle feeding of complete structure
        if self.feed_method == 'complete':
            entry_list.append(entry)
            atomtype_list.append(atomtypes)
            membership_list.append(memberships)
            adjacency_list.append(adjacency)
            target_list.append(target)

        # Handle single chain and pairwise feedings
        else:
            
            # Handle single chain feeding
            if self.feed_method == 'single_chain':
                
                # Get all single chain IDs and add to list
                for chain in unique_chains:
                    chain_combos.append([chain])

            # Handle pairwise feeding
            elif self.feed_method == 'pairwise':
                
                # Generate all pairwise chain combinations and add to list
                chain_combos.extend(combinations(unique_chains, 2))

            # Raise an error for unknown feed method
            else:
                raise(ValueError('Unknown feed method "{}".'.format(self.feed_method)))

            # Loop through each subset selection and add the matrices to the appropriate list
            for chains in chain_combos:
                selection = select_chains(atomtypes, memberships, adjacency, target, chains)

                entry_list.append('{}_'.format(entry, '_'.join([str(i) for i in chains])))
                atomtype_list.append(selection[0])
                membership_list.append(selection[1])
                adjacency_list.append(selection[2])
                target_list.append(selection[3])

        # Create the data dictionary
        data_dict = {
            'id': entry_list,
            'atomtypes': atomtype_list,
            'memberships': membership_list,
            'adjacency': adjacency_list,
            'target': target_list
        }

        # Return the data for training
        return data_dict
