import torch
from torch.utils.data import Dataset
import numpy as np
import periodictable as pt

from .data_utils import *

class TesselateDataset(Dataset):
    """
    Dataset class for structural data.
    
    Args:
    - accession_list (str) - File path from which to read PDB IDs for dataset.
    - graph_dir (str) - Directory containing the nodes, edges, and mask files.
    - contacts_dir (str) - Directory containing the .contacts files from
        get_contacts.py.
    - return_data (list) - List of datasets to return. Value must be 'all' or
        a subset of the following list:
            - pdb_id
            - model
            - atom_nodes
            - atom_adj
            - atom_contact
            - atom_mask
            - res_adj
            - res_contact
            - res_mask
            - mem_mat
            - idx2atom_dict
            - idx2res_dict    
    """
    
    def __init__(self, accession_list, graph_dir, contacts_dir, add_covalent=False, return_data='all'):
        
        if return_data == 'all':
            self.return_data = [
                'pdb_id',
                'model',
                'atom_nodes',
                'atom_adj',
                'atom_contact',
                'atom_mask',
                'res_adj',
                'res_contact',
                'res_mask',
                'mem_mat',
                'idx2atom_dict',
                'idx2res_dict'
            ]
        
        # Store reference to accession list file
        self.accession_list = accession_list
        
        # Store references to the necessary directories
        self.graph_dir = graph_dir
        self.contacts_dir = contacts_dir
        
        # Whether to add covalent bonds to prediction task and
        # remove sequence non-deterministic covalent bonds from the adjacency matrix
        self.add_covalent=add_covalent
        
        # Read in and store a list of accession IDs
        with open(accession_list, 'r') as handle:
            self.accessions = np.array([acc.strip().lower().split() for acc in handle.readlines()])

            
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
        - Dictionary of dataset example. All tensors are sparse when possible.
        """
        
        # initialize the return dictionary
        return_dict = {}
        
        acc_entry = self.accessions[idx]
        
        # Get the PDB ID
        acc = acc_entry[0]

        # Get the model number if one exists
        if len(acc_entry) == 1:
            model = 1
        else:
            model = acc_entry[1]
            
        # Read and process the files
        data = read_files(acc, model, self.graph_dir, self.contacts_dir)
        data = process_res_data(data)
        
        # Generate the mapping dictionaries
        atom2idx_dict, idx2atom_dict = get_map_dicts(data['atom_nodes']['atom'].unique())
        res2idx_dict, idx2res_dict = get_map_dicts(data['res_nodes']['res'].unique())
        
        # Get numbers of atoms and residues per sample
        n_atoms = len(atom2idx_dict)
        n_res = len(res2idx_dict)
        
        # Handle all of the possible returned datasets
        if 'pdb_id' in self.return_data:
            return_dict['pdb_id'] = acc
            
        if 'model' in self.return_data:
            return_dict['model'] = model
            
        if 'atom_nodes' in self.return_data:
            ele_nums = [pt.elements.symbol(element).number for element in data['atom_nodes']['element']]
            return_dict['atom_nodes'] = torch.LongTensor(ele_nums)
        
        if 'atom_adj' in self.return_data:
            adj = create_adj_mat(data['atom_edges'], atom2idx_dict, mat_type='atom_graph').T
            
            x = torch.LongTensor(adj[0, :]).squeeze()
            y = torch.LongTensor(adj[1, :]).squeeze()
            val = torch.FloatTensor(adj[2, :]).squeeze()
            
            atom_adj = torch.zeros([n_atoms, n_atoms]).index_put_((x, y), val, accumulate=False)
            
            atom_adj = atom_adj.index_put_((y, x), val, accumulate=False)
            
            atom_adj[range(n_atoms), range(n_atoms)] = 1
            
            atom_adj = (atom_adj > 0).float()
            
            return_dict['atom_adj'] = atom_adj
            
        if 'atom_contact' in self.return_data:
            atom_contact = create_adj_mat(data['atom_contact'], atom2idx_dict, mat_type='atom_contact').T
            
            x = torch.LongTensor(atom_contact[0, :]).squeeze()
            y = torch.LongTensor(atom_contact[1, :]).squeeze()
            z = torch.LongTensor(atom_contact[2, :]).squeeze()

            atom_contact = torch.zeros([n_atoms, n_atoms, 8]).index_put_((x, y, z),
                                                                    torch.ones(len(x)))
            atom_contact = atom_contact.index_put_((y, x, z), 
                                                   torch.ones(len(x)))
            
            return_dict['atom_contact'] = atom_contact
            
        if 'atom_mask' in self.return_data:
            atom_mask = create_idx_list(data['atom_mask'], atom2idx_dict)
            
            masked_pos = torch.from_numpy(atom_mask)
            
            if self.add_covalent:
                channels = 9
            else:
                channels = 8
            
            mask = torch.ones([n_atoms, n_atoms, channels])
            mask[masked_pos, :, :] = 0
            mask[:, masked_pos, :] = 0
            
            return_dict['atom_mask'] = mask
            
        if 'res_adj' in self.return_data:            
            adj = create_adj_mat(data['res_edges'], res2idx_dict, mat_type='res_graph').T
            
            x = torch.LongTensor(adj[0, :]).squeeze()
            y = torch.LongTensor(adj[1, :]).squeeze()
            
            res_adj = torch.zeros([n_res, n_res]).index_put_((x, y), torch.ones(len(x)))
            
            res_adj = res_adj.index_put_((y, x), torch.ones(len(x)))
            
            res_adj[range(n_res), range(n_res)] = 1
            
            return_dict['res_adj'] = res_adj
            
        if 'res_contact' in self.return_data:
            res_contact = create_adj_mat(data['res_contact'], res2idx_dict, mat_type='res_contact').T
            
            x = torch.LongTensor(res_contact[0, :]).squeeze()
            y = torch.LongTensor(res_contact[1, :]).squeeze()
            z = torch.LongTensor(res_contact[2, :]).squeeze()

            res_contact = torch.zeros([n_res, n_res, 8]).index_put_((x, y, z),
                                                                    torch.ones(len(x)))
            
            res_contact = res_contact.index_put_((y, x, z),
                                                 torch.ones(len(x)))
            
            return_dict['res_contact'] = res_contact
            
        if 'res_mask' in self.return_data:
            res_mask = create_idx_list(data['res_mask'], res2idx_dict)
            
            masked_pos = torch.from_numpy(res_mask)
            
            if self.add_covalent:
                channels = 9
            else:
                channels = 8
            
            mask = torch.ones([n_res, n_res, channels])
            mask[masked_pos, :, :] = 0
            mask[:, masked_pos, :] = 0
            
            return_dict['res_mask'] = mask
            
        if 'mem_mat' in self.return_data:
            mem_mat = create_mem_mat(atom2idx_dict, res2idx_dict).T
            
            x = torch.LongTensor(mem_mat[0, :]).squeeze()
            y = torch.LongTensor(mem_mat[1, :]).squeeze()
            
            mem_mat = torch.zeros([n_res, n_atoms]).index_put_((x, y),
                                                               torch.ones(len(x)))
            
            return_dict['mem_mat'] = mem_mat
            
        if 'idx2atom_dict' in self.return_data:
            return_dict['idx2atom_dict'] = idx2atom_dict
            
        if 'idx2res_dict' in self.return_data:
            return_dict['idx2res_dict'] = idx2res_dict
            
        # Return the processed data
        return return_dict
