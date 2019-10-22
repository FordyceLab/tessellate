import pandas as pd
import ntpath
import networkx as nx
import sys

# molreport extraction
def extract_molreport(filepath, strip_H=False):
    atom_info = {
        'atom': [],
        'element': [],
        'type': [],
        'hyb': [],
        'charge': []
    }
    
    bond_info = {
        'start': [],
        'end': [],
        'order': []
    }
    
    elements = {}
    
    with open(filepath) as molreport:
        for line in molreport.readlines():
            if line.startswith('ATOM'):
                splitline = line.strip().split()
                
                if (splitline[2] != 'H' and strip_H) or not strip_H:
                    atom_info['atom'].append(int(splitline[1]))
                    atom_info['element'].append(splitline[2])
                    atom_info['type'].append(splitline[4])
                    atom_info['hyb'].append(int(splitline[6]))
                    atom_info['charge'].append(float(splitline[8]))
                
                elements[int(splitline[1])] = splitline[2]
                
            elif line.startswith('BOND'):
                splitline = line.strip().split()
                
                bond_start = int(splitline[3])
                bond_end = int(splitline[5])
                
                not_H = elements[bond_start] != 'H' and elements[bond_end] != 'H'
                
                if (not_H and strip_H) or not strip_H:
                
                    if bond_start < bond_end:
                        bond_info['start'].append(bond_start)
                        bond_info['end'].append(bond_end)

                    else:
                        bond_info['start'].append(bond_end)
                        bond_info['end'].append(bond_start)
                
                    bond_info['order'].append(splitline[7])
                
    return (pd.DataFrame(atom_info), pd.DataFrame(bond_info))


def extract_ids(filepath):
    ids = pd.read_table(filepath, names=['atom', 'identifier', 'element'])
    return ids


# Merge id info into molreport
def merge_molreport_ids(molreport_atoms, molreport_bonds, ids):
    
    # Handle atoms file
    atom_out = pd.merge(molreport_atoms, ids, on=['atom', 'element']).drop('atom',
                                           axis=1).rename(columns={'identifier': 'atom'})
    atom_out = atom_out[['atom', 'element', 'type', 'hyb', 'charge']]
    
    # Handle bonds
    start_merge = pd.merge(molreport_bonds,
         ids[['atom', 'identifier']],
         left_on='start',
         right_on='atom').drop(['start', 'atom'],
                               axis=1).rename(columns={'identifier': 'start'})

    end_merge = pd.merge(start_merge,
                         ids[['atom', 'identifier']],
                         left_on='end',
                         right_on='atom').drop(['end', 'atom'],
                                               axis=1).rename(columns={'identifier': 'end'})

    bond_out = end_merge[['start', 'end', 'order']]
    
    return (atom_out, bond_out)


def strip_hydrogen(atoms, bonds):
    atoms = atoms[atoms['element'] != 'H']
    
    bonds = bonds[bonds['start'].isin(atoms['atom']) & bonds['end'].isin(atoms['atom'])]
    
    return (atoms, bonds)


def augment_bonds(bonds):
    start_info = bonds['start'].str.split(':', expand=True).rename(columns={0: 'start_chain',
                                                                        1: 'start_res',
                                                                        2: 'start_num',
                                                                        3: 'start_atom'})

    end_info = bonds['end'].str.split(':', expand=True).rename(columns={0: 'end_chain',
                                                                        1: 'end_res',
                                                                        2: 'end_num',
                                                                        3: 'end_atom'})

    bonds = pd.concat([bonds, start_info, end_info], axis=1)
    
    bonds['start_num'] = bonds['start_num'].astype(int)
    bonds['end_num'] = bonds['end_num'].astype(int)
    
    return bonds


def augment_atoms(atoms):
    atoms_info = atoms['atom'].str.split(':', expand=True).rename(columns={0: 'chain',
                                                                           1: 'res',
                                                                           2: 'num',
                                                                           3: 'atom_name'})
    
    atoms = pd.concat([atoms, atoms_info], axis=1)
    
    atoms['num'] = atoms['num'].astype(int)
    
    return atoms


def identify_gaps(chain_atoms):
    min_num = chain_atoms['num'].min()
    max_num = chain_atoms['num'].max()
    
    present = []
    absent = []
    breakpoints = []
    
    unique_idxs = chain_atoms['num'].unique()
    
    for i in range(min_num, max_num + 1):
        if i in unique_idxs:
            present.append(i)
            
            term = i == min_num or i == max_num
            up_break = i + 1 not in chain_atoms['num']
            down_break = i - 1 not in chain_atoms['num']
            
            breakpoint = not term and (up_break or down_break)
            
            if breakpoint:
                breakpoints.append(i)
                
        else:
            absent.append(i)
            
    return (present, absent, breakpoints)


def patch_gaps(chain_atoms, chain_bonds, seq_atoms, seq_bonds, absent, breakpoints):
    all_missing = []
    
    # Get chain ID
    chain = chain_atoms['chain'].unique()[0]
    
    # Get missing atoms and bonds
    missing_atoms = seq_atoms[(seq_atoms['chain'] == chain) &
                              (seq_atoms['num'].isin(absent)) &
                              (~seq_atoms['atom'].isin(chain_atoms['atom']))]
    
    missing_bonds = seq_bonds[seq_bonds['start'].isin(missing_atoms['atom']) | seq_bonds['end'].isin(missing_atoms['atom'])]
    
    chain_atoms = pd.concat([chain_atoms, missing_atoms])
    chain_bonds = pd.concat([chain_bonds, missing_bonds])
    
    all_missing.append(missing_atoms)
    
    # Check if missing atoms alone complete the chain
    G = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')
    
    # If still not connected, merge breakpoint residues
    if not nx.is_connected(G):
        # Get missing atoms and bonds
        missing_atoms = seq_atoms[(seq_atoms['chain'] == chain) & (seq_atoms['num'].isin(breakpoints))]
        
        # Select only atoms with the same residue name
        missing_atoms = pd.merge(chain_atoms[['chain', 'res', 'num']], missing_atoms, how='inner', on=['chain', 'res', 'num'])
        missing_atoms = missing_atoms[~missing_atoms['atom'].isin(chain_atoms)]
        all_missing.append(missing_atoms)
        
        # Get missing bonds
        missing_bonds = seq_bonds[seq_bonds['start'].isin(missing_atoms['atom']) | seq_bonds['end'].isin(missing_atoms['atom'])]

        # Add missing bonds and atoms
        chain_atoms = pd.concat([chain_atoms, missing_atoms], axis=0, sort=False)
        
        # Add missing bonds
        missing_bonds = seq_bonds[seq_bonds['start'].isin(chain_atoms['atom']) & \
                                      seq_bonds['end'].isin(chain_atoms['atom'])]
        chain_bonds = pd.merge(chain_bonds, missing_bonds, how='outer', on=list(chain_bonds.columns))

        # Check if missing atoms alone complete the chain
        G = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')
        
        if not nx.is_connected(G):
           
            missing_atoms = seq_atoms[(seq_atoms['chain'] == chain) &
                                      (seq_atoms['num'].isin(chain_atoms['num'])) &
                                      (~seq_atoms['atom'].isin(chain_atoms['atom']))]


            chain_atoms = pd.concat([chain_atoms, missing_atoms], axis=0, sort=False)

            # Get missing bonds
            missing_bonds = seq_bonds[seq_bonds['start'].isin(chain_atoms['atom']) | \
                                      seq_bonds['end'].isin(chain_atoms['atom'])]

            chain_bonds = pd.concat([chain_bonds, missing_bonds], axis=0, sort=False)

            all_missing.append(missing_atoms)

            G = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')
        
    assert nx.is_connected(G)
    
    missing_atoms = pd.concat(all_missing, axis=0, sort=False)
    
    return (chain_atoms, chain_bonds, list(missing_atoms['atom'].unique()))


def fill_gaps(seq_atoms, seq_bonds, struct_atoms, struct_bonds):

    unique_chains = struct_atoms['chain'].unique()
    
    atoms = []
    bonds = []
    masked_atoms = []
    
    for chain in unique_chains:
        
        chain_atoms = struct_atoms[(struct_atoms['chain'] == chain)]

        chain_bonds = struct_bonds[(struct_bonds['start_chain'] == chain) | (struct_bonds['end_chain'] == chain)]
        
        G = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')

        if nx.is_connected(G):
            atoms.append(chain_atoms)
            bonds.append(chain_bonds)
            
        else:
            present, absent, breakpoints = identify_gaps(chain_atoms)
            
            # Add any disconnects not found by residue gap check
            breakpoints.extend(check_res_connections(chain_bonds))
            
            clean_atoms, clean_bonds, added_atoms = patch_gaps(chain_atoms, chain_bonds, seq_atoms, seq_bonds, absent, breakpoints)
            
            atoms.append(clean_atoms)
            bonds.append(clean_bonds)
            masked_atoms.extend(added_atoms)

    out_atoms = pd.concat(atoms).drop_duplicates()
    out_bonds = pd.concat(bonds).drop_duplicates()
            
    return (out_atoms, out_bonds, masked_atoms)


def check_res_connections(bonds):
    disconnects = []
    
    simple_bonds = bonds[['start_chain', 'start_num', 'end_chain', 'end_num']].drop_duplicates()
    
    chains = sorted(list(set(list(simple_bonds['start_chain'].unique()) +
                             list(simple_bonds['end_chain'].unique()))))
    
    for chain in chains:
        
        # Get residue-residue bonds
        chain_bonds = simple_bonds.loc[simple_bonds['start_chain'] == chain, ['start_num', 'end_num']].drop_duplicates()
        chain_bonds = chain_bonds.loc[simple_bonds['start_num'] != simple_bonds['end_num']]
        
        # Extract all of the connections
        cxns = [sorted([chain_bonds.iloc[i, 0], chain_bonds.iloc[i, 1]]) for i in range(len(chain_bonds))]
        
        res_nums = list(chain_bonds['start_num']) + list(chain_bonds['end_num'])
        max_num = max(res_nums)
        min_num = min(res_nums)
        
        for i in range(min_num, max_num):
            if [i, i+1] not in cxns:
                disconnects.extend([i, i+1])
    
    return disconnects


if __name__ == '__main__':
    
    nodes_out = sys.argv[1]
    edges_out = sys.argv[2]
    mask_out = sys.argv[3]
    
    files = sys.argv[4:]
    
    filesplits = {
        'struct': {},
        'seq': {}
    }

    for file in files:
        base, ext = ntpath.basename(file).split('.')
        suffix = base.split('_', 1)[1]

        if suffix == 'no_solvent':
            filesplits['struct'][ext] = file
        else:
            if suffix not in filesplits['seq']:
                filesplits['seq'][suffix] = {}

            filesplits['seq'][suffix][ext] = file

    # Handle the sequence
    indy_chains = []

    for key in filesplits['seq']:
        atoms, bonds = extract_molreport(filesplits['seq'][key]['molreport'])
        ids = extract_ids(filesplits['seq'][key]['id'])

        indy_chains.append(merge_molreport_ids(atoms, bonds, ids))

    seq_atoms = pd.concat([chain[0] for chain in indy_chains])
    seq_bonds = pd.concat([chain[1] for chain in indy_chains])

    seq_atoms, seq_bonds = strip_hydrogen(seq_atoms, seq_bonds)

    # Handle the structure
    atoms, bonds = extract_molreport(filesplits['struct']['molreport'])
    ids = extract_ids(filesplits['struct']['id'])

    struct_atoms, struct_bonds = merge_molreport_ids(atoms, bonds, ids)
    struct_atoms, struct_bonds = strip_hydrogen(struct_atoms, struct_bonds)

    seq_atoms = augment_atoms(seq_atoms)
    struct_atoms = augment_atoms(struct_atoms)

    seq_bonds = augment_bonds(seq_bonds)
    struct_bonds = augment_bonds(struct_bonds)

    out_atoms, out_bonds, masked_atoms = fill_gaps(seq_atoms, seq_bonds, struct_atoms, struct_bonds)
    
    out_atoms = out_atoms.sort_values(['chain', 'num'], ascending=[True, True])
    out_atoms = out_atoms[['atom', 'element']].drop_duplicates()
    
    out_atoms.to_csv(nodes_out, index=False, columns=['atom', 'element'])
    out_bonds.to_csv(edges_out, index=False, columns=['start', 'end', 'order'])
    
    with open(mask_out, 'w') as maskfile:
        maskfile.write("\n".join(masked_atoms))
    