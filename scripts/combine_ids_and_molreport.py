"""Module and script to combine IDs with molreports to form graphs and masks.

This module provides functions and a script to extract bond and atom identifier
information, combine these IDs with bonds from the molreport file, and output:

1) An atom-level node list
2) An atom-level covalent bond list
3) A mask of atoms that have been added to the graph, but are not present in the
    structure
"""

import ntpath
import sys
import pandas as pd
import networkx as nx

# molreport extraction
def extract_molreport(filepath, strip_h=False):
    """
    Extract relevant information from molreport file type.

    Args:
    - filepath (str) - Path to molreport file.
    - strip_h (bool) - Whether to strip hydrogens from molreport file
        (default=False).

    Returns:
    - Tuple of Pandas dataframes, one for atom information and one for bond
        information.
    """

    # Dict to hold the atom information
    atom_info = {
        'atom': [],
        'element': [],
        'type': [],
        'hyb': [],
        'charge': []
    }

    # Dict to hold the bond information
    bond_info = {
        'start': [],
        'end': [],
        'order': []
    }

    # Dict to hold the element identities
    elements = {}

    # Open the molreport file
    with open(filepath) as molreport:

        # Read the file
        for line in molreport.readlines():

            # Handle atoms case
            if line.startswith('ATOM'):

                # Split the line
                splitline = line.strip().split()

                # Extract relevant information for each atom, respecting
                # hydrogen stripping parameters
                if (splitline[2] != 'H' and strip_h) or not strip_h:
                    atom_info['atom'].append(int(splitline[1]))
                    atom_info['element'].append(splitline[2])
                    atom_info['type'].append(splitline[4])
                    atom_info['hyb'].append(int(splitline[6]))
                    atom_info['charge'].append(float(splitline[8]))

                # Get the element identity
                elements[int(splitline[1])] = splitline[2]

            # Handle bonds case
            elif line.startswith('BOND'):

                # Split the line
                splitline = line.strip().split()

                # Get the bond start and end points
                bond_start = int(splitline[3])
                bond_end = int(splitline[5])

                # Whether bond includes hydrogen
                not_h = (elements[bond_start] != 'H' and
                         elements[bond_end] != 'H')

                # Extract relevant information for each atom, respecting
                # hydrogen stripping parameters
                if (not_h and strip_h) or not strip_h:

                    # Extract bond info, with correct ordering
                    if bond_start < bond_end:
                        bond_info['start'].append(bond_start)
                        bond_info['end'].append(bond_end)
                    else:
                        bond_info['start'].append(bond_end)
                        bond_info['end'].append(bond_start)

                    # Extract bond order (e.g., single, double, etc.)
                    bond_info['order'].append(splitline[7])

    # Return a data frame of the relevant info
    atom_info = pd.DataFrame(atom_info)
    atom_info['element'] = atom_info['element'].apply(lambda x: x.title())

    bond_info = pd.DataFrame(bond_info)

    return (atom_info, bond_info)


def extract_ids(filepath):
    """
    Extract atom identying attributes from file.

    Args:
    - filepath (str) - Path to ID file.

    Returns:
    - Pandas DataFrame of the atom name, identifier, and element.
    """

    ids = pd.read_table(filepath, names=['atom', 'identifier', 'element'])
    ids['element'] = ids['element'].apply(lambda x: x.title())
    return ids


def merge_molreport_ids(molreport_atoms, molreport_bonds, ids):
    """Merge molreport with ID information.

    Merges ID information (chain, residues, atom, etc.) with molreport
    information including bonds.

    Args:
    - molreport_atoms (pd.DataFrame) - Pandas DataFrame containing all atom
        information from the molreport.
    - molreport_bonds (pd.DataFrame) - Pandas DataFrame containing all bond
        information from the molreport.
    - ids (pd.DataFrame) - Pandas DataFrame containing indentifying information
        for each individual atom, to be joined into less descriptive molreport
        identifiers.

    Returns:
    - Tuple of Pandas DataFrames (atoms and bonds, respoectively) with merged ID
        information for each atom.
    """

    # Handle atoms file
    atom_out = (
        pd.merge(molreport_atoms, ids, on=['atom', 'element'])
        .drop('atom', axis=1)
        .rename(columns={'identifier': 'atom'})
    )

    atom_out = atom_out[['atom', 'element', 'type', 'hyb', 'charge']]

    # Handle bonds
    start_merge = (
        pd.merge(molreport_bonds,
                 ids[['atom', 'identifier']],
                 left_on='start',
                 right_on='atom')
        .drop(['start', 'atom'], axis=1)
        .rename(columns={'identifier': 'start'})
    )

    end_merge = (
        pd.merge(start_merge,
                 ids[['atom', 'identifier']],
                 left_on='end',
                 right_on='atom')
        .drop(['end', 'atom'], axis=1)
        .rename(columns={'identifier': 'end'})
    )

    bond_out = end_merge[['start', 'end', 'order']]

    return (atom_out, bond_out)


def strip_hydrogen(atoms, bonds):
    """
    Remove hydrogens from the atom and bond tables.
    """

    atoms = atoms[atoms['element'] != 'H']

    bonds = bonds[bonds['start'].isin(atoms['atom']) &
                  bonds['end'].isin(atoms['atom'])]

    return (atoms, bonds)


def augment_bonds(bonds):
    """
    Split bond identifiers into component columns.
    """

    start_info = (
        bonds['start'].str.split(':', expand=True)
        .rename(columns={0: 'start_chain',
                         1: 'start_res',
                         2: 'start_num',
                         3: 'start_atom'})
    )

    end_info = (
        bonds['end'].str.split(':', expand=True)
        .rename(columns={0: 'end_chain',
                         1: 'end_res',
                         2: 'end_num',
                         3: 'end_atom'})
    )

    bonds = pd.concat([bonds, start_info, end_info], axis=1)

    bonds['start_num'] = bonds['start_num'].astype(int)
    bonds['end_num'] = bonds['end_num'].astype(int)

    return bonds


def augment_atoms(atoms):
    """
    Split atom identifiers into component columns.
    """

    atoms_info = (
        atoms['atom'].str.split(':', expand=True)
        .rename(columns={0: 'chain',
                         1: 'res',
                         2: 'num',
                         3: 'atom_name'})
    )

    atoms = pd.concat([atoms, atoms_info], axis=1)

    atoms['num'] = atoms['num'].astype(int)

    return atoms


def identify_gaps(chain_atoms):
    """
    Identify gaps in chain of atoms.
    """

    min_num = chain_atoms['num'].min()
    max_num = chain_atoms['num'].max()

    present = []
    absent = []
    breakpoints = []

    unique_idxs = chain_atoms['num'].unique()

    for i in range(min_num, max_num + 1):
        if i in unique_idxs:
            present.append(i)

            term = i in (min_num, max_num)
            up_break = i + 1 not in chain_atoms['num']
            down_break = i - 1 not in chain_atoms['num']

            breakpoint = not term and (up_break or down_break)

            if breakpoint:
                breakpoints.append(i)

        else:
            absent.append(i)

    return (present, absent, breakpoints)


def patch_gaps(chain, seq, absent, breakpoints):
    """
    Patch gaps in a chain.
    """

    # Extract information
    chain_atoms, chain_bonds = chain
    seq_atoms, seq_bonds = seq

    # Initialize a list for the missing atoms
    all_missing = []

    # Get chain ID
    chain = chain_atoms['chain'].unique()[0]

    # Get missing atoms and bonds
    missing_atoms = seq_atoms[(seq_atoms['chain'] == chain) &
                              (seq_atoms['num'].isin(absent)) &
                              (~seq_atoms['atom'].isin(chain_atoms['atom']))]

    missing_bonds = seq_bonds[seq_bonds['start'].isin(missing_atoms['atom']) |
                              seq_bonds['end'].isin(missing_atoms['atom'])]

    chain_atoms = pd.concat([chain_atoms, missing_atoms])
    chain_bonds = pd.concat([chain_bonds, missing_bonds])

    all_missing.append(missing_atoms)

    # Check if missing atoms alone complete the chain
    graph = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')

    # If still not connected, merge breakpoint residues
    if not nx.is_connected(graph):
        # Get missing atoms and bonds
        missing_atoms = seq_atoms[(seq_atoms['chain'] == chain) &
                                  (seq_atoms['num'].isin(breakpoints))]

        # Select only atoms with the same residue name
        missing_atoms = pd.merge(chain_atoms[['chain', 'res', 'num']],
                                 missing_atoms, how='inner',
                                 on=['chain', 'res', 'num'])

        missing_atoms = missing_atoms[~missing_atoms['atom'].isin(chain_atoms)]
        all_missing.append(missing_atoms)

        # Get missing bonds
        missing_bonds = (
            seq_bonds[seq_bonds['start'].isin(missing_atoms['atom']) |
                      seq_bonds['end'].isin(missing_atoms['atom'])]
        )

        # Add missing bonds and atoms
        chain_atoms = pd.concat([chain_atoms, missing_atoms], axis=0,
                                sort=False)

        # Add missing bonds
        missing_bonds = seq_bonds[seq_bonds['start'].isin(chain_atoms['atom']) &
                                  seq_bonds['end'].isin(chain_atoms['atom'])]

        chain_bonds = pd.merge(chain_bonds, missing_bonds, how='outer',
                               on=list(chain_bonds.columns))

        # Check if missing atoms alone complete the chain
        graph = nx.from_pandas_edgelist(chain_bonds, source='start', target='end')

        if not nx.is_connected(graph):

            missing_atoms = (
                seq_atoms[(seq_atoms['chain'] == chain) &
                          (seq_atoms['num'].isin(chain_atoms['num'])) &
                          (~seq_atoms['atom'].isin(chain_atoms['atom']))]
            )


            chain_atoms = pd.concat([chain_atoms, missing_atoms], axis=0,
                                    sort=False)

            # Get missing bonds
            missing_bonds = (
                seq_bonds[seq_bonds['start'].isin(chain_atoms['atom']) |
                          seq_bonds['end'].isin(chain_atoms['atom'])]
            )

            chain_bonds = pd.concat([chain_bonds, missing_bonds], axis=0,
                                    sort=False)

            all_missing.append(missing_atoms)

            graph = nx.from_pandas_edgelist(chain_bonds, source='start',
                                            target='end')

    assert nx.is_connected(graph)

    missing_atoms = pd.concat(all_missing, axis=0, sort=False)

    return (chain_atoms, chain_bonds, list(missing_atoms['atom'].unique()))


def fill_gaps(seq_atoms, seq_bonds, struct_atoms, struct_bonds):
    """
    Fill gaps in a chain with the required atoms and bonds from the sequence.
    """

    unique_chains = struct_atoms['chain'].unique()

    atoms = []
    bonds = []
    masked_atoms = []

    for chain in unique_chains:

        chain_atoms = struct_atoms[(struct_atoms['chain'] == chain)]

        chain_bonds = struct_bonds[(struct_bonds['start_chain'] == chain) &
                                   (struct_bonds['end_chain'] == chain)]

        graph = nx.from_pandas_edgelist(chain_bonds,
                                        source='start',
                                        target='end')

        if nx.is_connected(graph):
            atoms.append(chain_atoms)
            bonds.append(chain_bonds)

        else:
            gaps = identify_gaps(chain_atoms)

            # Add any disconnects not found by residue gap check
            gaps[2].extend(check_res_connections(chain_bonds))

            cleaned = patch_gaps(chain=(chain_atoms, chain_bonds),
                                 seq=(seq_atoms, seq_bonds),
                                 absent=gaps[1], breakpoints=gaps[2])

            atoms.append(cleaned[0])
            bonds.append(cleaned[1])
            masked_atoms.extend(cleaned[2])

    atoms = pd.concat(atoms).drop_duplicates()
    bonds = pd.concat(bonds).drop_duplicates()

    return (atoms, bonds, masked_atoms)


def check_res_connections(bonds):
    """
    Check the residue connections.
    """

    disconnects = []
    simple_bonds = (
        bonds[['start_chain', 'start_num', 'end_chain', 'end_num']]
        .drop_duplicates()
    )

    chains = sorted(list(set(list(simple_bonds['start_chain'].unique()) +
                             list(simple_bonds['end_chain'].unique()))))

    for chain in chains:

        # Get residue-residue bonds
        chain_bonds = (
            simple_bonds.loc[simple_bonds['start_chain'] == chain,
                             ['start_num', 'end_num']]
            .drop_duplicates()
        )

        chain_bonds = chain_bonds.loc[simple_bonds['start_num'] !=
                                      simple_bonds['end_num']]

        # Extract all of the connections
        cxns = [sorted([chain_bonds.iloc[i, 0], chain_bonds.iloc[i, 1]])
                for i in range(len(chain_bonds))]

        res_nums = list(chain_bonds['start_num']) + list(chain_bonds['end_num'])

        max_num = max(res_nums)
        min_num = min(res_nums)

        for i in range(min_num, max_num):
            if [i, i+1] not in cxns:
                disconnects.extend([i, i+1])

    return disconnects


def combine_ids_and_molreport(filesplits):
    """
    Combine the atom ids with the molreport information.
    """

    ########################################
    # Handle the sequence file information #
    ########################################

    # Make a list of the independent chains
    indy_chains = []

    # Get the chain sequences from all of the sequence files
    for key in filesplits['seq']:
        atoms, bonds = extract_molreport(filesplits['seq'][key]['molreport'])
        ids = extract_ids(filesplits['seq'][key]['id'])

        indy_chains.append(merge_molreport_ids(atoms, bonds, ids))

    # Get all of the atom and bond IDs from the independent chains
    seq_atoms = pd.concat([chain[0] for chain in indy_chains])
    seq_bonds = pd.concat([chain[1] for chain in indy_chains])

    # Strip the hydrogens
    seq_atoms, seq_bonds = strip_hydrogen(seq_atoms, seq_bonds)

    #########################################
    # Handle the structure file information #
    #########################################

    # Get the structure information from the molreport file
    atoms, bonds = extract_molreport(filesplits['struct']['molreport'])
    ids = extract_ids(filesplits['struct']['id'])

    # Merge the structure atoms with their IDs and strip hydrogens
    struct_atoms, struct_bonds = merge_molreport_ids(atoms, bonds, ids)
    struct_atoms, struct_bonds = strip_hydrogen(struct_atoms, struct_bonds)

    # Augment the atoms with their identifying columns
    seq_atoms = augment_atoms(seq_atoms)
    struct_atoms = augment_atoms(struct_atoms)

    # Augment the bonds with their identifying columns
    seq_bonds = augment_bonds(seq_bonds)
    struct_bonds = augment_bonds(struct_bonds)

    # Fill gaps in the graph
    out_atoms, out_bonds, masked_atoms = fill_gaps(seq_atoms,
                                                   seq_bonds,
                                                   struct_atoms,
                                                   struct_bonds)

    return (out_atoms, out_bonds, masked_atoms)


if __name__ == '__main__':

    # Get the output file names as the incoming command line arguments
    NODES_OUT = sys.argv[1]
    EDGES_OUT = sys.argv[2]
    MASK_OUT = sys.argv[3]

    # Get the input files as the trailing command line arguments
    FILES = sys.argv[4:]

    # Dict to hold splits of filetypes
    FILESPLITS = {
        'struct': {},
        'seq': {}
    }

    # Run through each file and determing the type
    for file in FILES:

        # Split the filename on extension
        base, ext = ntpath.basename(file).split('.')
        suffix = base.split('_', 1)[1]

        # Get structure files
        if suffix == 'no_solvent':
            FILESPLITS['struct'][ext] = file

        # Get sequence files
        else:
            if suffix not in FILESPLITS['seq']:
                FILESPLITS['seq'][suffix] = {}

            FILESPLITS['seq'][suffix][ext] = file

    # Run the main function to get all the combined information
    OUT_ATOMS, OUT_BONDS, MASKED_ATOMS = combine_ids_and_molreport(FILESPLITS)

    # Sort the atoms and drop duplicate values based on atom and element labels
    OUT_ATOMS = OUT_ATOMS.sort_values(['chain', 'num'], ascending=[True, True])
    OUT_ATOMS = OUT_ATOMS[['atom', 'element']].drop_duplicates()

    # Write CSVs of data
    OUT_ATOMS.to_csv(NODES_OUT, index=False, columns=['atom', 'element'])
    OUT_BONDS.to_csv(EDGES_OUT, index=False, columns=['start', 'end', 'order'])

    # Write the masked atoms
    with open(MASK_OUT, 'w') as maskfile:
        maskfile.write('\n'.join(MASKED_ATOMS))
