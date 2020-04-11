import numpy as np
import os
import pandas as pd


def read_files(acc, model, graph_dir, contacts_dir):
    """
    Read graph and contacts files.
    
    Args:
    - acc (str) - String of the PDB ID (lowercese).
    - model (int) - Model number of the desired bioassembly.
    - graph_dir (str) - Directory containing the nodes, edges,
        and mask files.
    - contacts_dir (str) - Directory containing the .contacts
        files from get_contacts.py.
        
    Returns:
    - Dictionary of DataFrames and lists corresponding to
        graph nodes, edges, mask, and contacts. 
    """

    # Get the file names for the graph files
    node_file = os.path.join(graph_dir, '{}-{}_nodes.csv'.format(acc, model))
    edge_file = os.path.join(graph_dir, '{}-{}_edges.csv'.format(acc, model))
    mask_file = os.path.join(graph_dir, '{}-{}_mask.csv'.format(acc, model))

    # Get the contacts file
    contacts_file = os.path.join(contacts_dir, '{}-{}.contacts'.format(acc, model))

    # Read the nodes and edges
    nodes = pd.read_csv(node_file)
    edges = pd.read_csv(edge_file)

    # Check if the mask is empty
    if os.path.getsize(mask_file) > 0:
        with open(mask_file) as f:
            mask = f.read().split('\n')
    else:
        mask = []

    # Read the contacts
    contacts = pd.read_table(contacts_file, sep='\t',
                             header=None, names=['type', 'start', 'end'])

    # Return the data
    data = {
        'nodes': nodes,
        'edges': edges,
        'mask': mask,
        'contacts': contacts
    }

    return data
    
    
def process_res_data(data):
    """
    Process residue-level data from atom-level data.
    
    Args:
    - data (dict) - Dictionary of graph data output from `read_files`.
    
    Returns:
    - Dictionary of atom and residue graph and contact data.
    """

    # Extract data form dict
    nodes = data['nodes']
    edges = data['edges']
    mask = data['mask']
    contacts = data['contacts']

    # Get residue nodes
    res_nodes = pd.DataFrame()
    res_nodes['res'] = [':'.join(atom.split(':')[:3]) for atom in nodes['atom']]
    res_nodes = res_nodes.drop_duplicates().reset_index(drop=True)

    # Get residue edges
    res_edges = edges.copy()
    res_edges['start'] = [':'.join(atom.split(':')[:3]) for atom in res_edges['start']]
    res_edges['end'] = [':'.join(atom.split(':')[:3]) for atom in res_edges['end']]
    res_edges = res_edges[res_edges['start'] != res_edges['end']].drop_duplicates().reset_index(drop=True)

    # Get residue contacts
    res_contacts = contacts.copy()
    res_contacts['start'] = [':'.join(atom.split(':')[:3]) for atom in res_contacts['start']]
    res_contacts['end'] = [':'.join(atom.split(':')[:3]) for atom in res_contacts['end']]
    res_contacts = res_contacts[res_contacts['start'] != res_contacts['end']].drop_duplicates().reset_index(drop=True)

    # Get residue mask
    res_mask = list(set([':'.join(atom.split(':')[:3]) for atom in mask]))

    # Return data dict
    data = {
        'atom_nodes': nodes,
        'atom_edges': edges,
        'atom_contact': contacts,
        'atom_mask': mask,
        'res_nodes': res_nodes,
        'res_edges': res_edges,
        'res_contact': res_contacts,
        'res_mask': res_mask
    }

    return data


def get_map_dicts(entity_list):
    """
    Map identifiers to indices and vice versa.
    
    Args:
    - entity_list (list) - List of entities (atoms, residues, etc.)
        to index.
    
    Returns:
    - Tuple of the entity to index and index to entity dicts, respectively.
    """
    
    # Create the entity:index dictionary
    ent2idx_dict = {entity: idx for idx, entity in enumerate(entity_list)}

    # Create the index:entity dictionary
    idx2ent_dict = {idx: entity for entity, idx in ent2idx_dict.items()}
    
    return (ent2idx_dict, idx2ent_dict)


def create_adj_mat(data, dict_map, mat_type):
    """
    Creates an adjacency matrix.
    
    Args:
    - data (DataFrame) - Dataframe with 'start' and 'end' column
        for each interaction. For atom-level adjacency, 'order' 
        column is also required. For atom or residue conatcts,
        'type' column is also required.
    
    Returns:
    - Coordinate format matrix (numpy). For atom adjacency, third column
        corresponds to bond order. For contacts, third column
        corresponds to channel.
    
    Channel mappings (shorthand from get_contacts.py source):

        0:
            hp             hydrophobic interactions
        1:
            hb             hydrogen bonds
            lhb            ligand hydrogen bonds
            hbbb           backbone-backbone hydrogen bonds
            hbsb           backbone-sidechain hydrogen bonds
            hbss           sidechain-sidechain hydrogen bonds
            hbls           ligand-sidechain residue hydrogen bonds
            hblb           ligand-backbone residue hydrogen bonds
        2:
            vdw            van der Waals
        3:
            wb             water bridges
            wb2            extended water bridges
            lwb            ligand water bridges
            lwb2           extended ligand water bridges
        4:
            sb             salt bridges
        5:
            ps             pi-stacking
        6:
            pc             pi-cation
        7:
            ts             t-stacking
    """
    
    # Initialize the coordinate list
    coord_mat = []

    # Map channel names to numeric channels
    channel = {
        # Hydrophobic interactions in first channel
        'hp': 0,

        # Hydrogen bonds in second channel
        'hb': 1,
        'lhb': 1, 
        'hbbb': 1,
        'hbsb': 1,
        'hbss': 1,
        'hbls': 1,
        'hblb': 1,

        # VdW in third channel
        'vdw': 2,

        # Water bridges
        'wb': 3, 
        'wb2': 3,
        'lwb': 3,
        'lwb2': 3,

        # Salt bridges
        'sb': 4,

        # Other interactions
        'ps': 5,
        'pc': 6,
        'ts': 7,
    }

    # Assemble the contacts
    for idx, row in data.iterrows():

        entry = [dict_map[row['start']], dict_map[row['end']]]

        # Add order or type if necessary
        if mat_type == 'atom_graph':
            entry.append(row['order'])
        elif mat_type == 'atom_contact':
            entry.append(channel[row['type']])
        elif mat_type == 'res_contact':
            entry.append(channel[row['type']])

        coord_mat.append(entry)

    return(np.array(coord_mat))


def create_mem_mat(atom_dict, res_dict):
    """
    Create a membership matrix mapping atoms to residues.
    
    Args:
    - atom_dict (dict) - Dictionary mapping atoms to indices.
    - res_dict (dict) - Dictionary mapping residues to indices.
    
    Returns:
    - Coordinate format membership matrix (numpy) with first
        row being residue number and the second column being
        atom number.
    """
    
    # Initialize the coordinate list
    mem_coord = []
    
    # Map atoms to residues
    for atom, atom_idx in atom_dict.items():
        res_idx = res_dict[':'.join(atom.split(':')[:3])]
        
        mem_coord.append([res_idx, atom_idx])
        
    mem_coord = np.array(mem_coord)
    
    return mem_coord


def create_idx_list(id_list, dict_map):
    """
    Create list of indices.
    
    Args:
    - id_list (list) - List of masked atom or residue identifiers.
    - dict_map (dict) - Dictionary mapping entities to indices.
    
    Returns:
    - A numpy array of the masked indices.
    """
    
    # Generate the numpy index array
    idx_array = np.array([dict_map[iden] for iden in id_list])
    
    return idx_array
