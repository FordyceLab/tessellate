from gemmi import cif
import pandas as pd
import sys
import ntpath

if __name__ == '__main__':
    
    filename = sys.argv[1]
    
    doc = cif.read_file(filename)
    
    block = doc.sole_block()

    # Map author chain IDs to sequences
    chain_ids = block.find_values('_entity_poly.pdbx_strand_id')
    chain_seqs = block.find_values('_entity_poly.pdbx_seq_one_letter_code_can')

    chain_seq_map = {}

    for idx, seq in zip(chain_ids, chain_seqs):
        clean_seq = seq.replace(';', '').replace('\n', '')

        for chain_id in idx.split(','):
            chain_seq_map[chain_id] = clean_seq
            
    # Extract atom info
    standard_res = list(block.find_values('_atom_site.label_comp_id'))
    standard_chain = list(block.find_values('_atom_site.label_asym_id'))
    standard_seq_pos = list(block.find_values('_atom_site.label_seq_id'))
    auth_res = list(block.find_values('_atom_site.auth_comp_id'))
    auth_chain = list(block.find_values('_atom_site.auth_asym_id'))
    auth_seq_pos = list(block.find_values('_atom_site.auth_seq_id'))
    ['standard_res', 'standard_chain', 'standard_seq_pos', 'auth_res', 'auth_chain', 'auth_seq_pos']

    data = {
        'standard_res': standard_res,
        'standard_chain': standard_chain,
        'standard_seq_pos': standard_seq_pos,
        'auth_res': auth_res,
        'auth_chain': auth_chain,
        'auth_seq_pos': auth_seq_pos
    }

    cif_atom_data = pd.DataFrame(data)
    
    # Extract embedded poly info
    standard_res = list(block.find_values('_pdbx_poly_seq_scheme.mon_id'))
    standard_chain = list(block.find_values('_pdbx_poly_seq_scheme.asym_id'))
    standard_seq_pos = list(block.find_values('_pdbx_poly_seq_scheme.seq_id'))
    auth_res = list(block.find_values('_pdbx_poly_seq_scheme.pdb_mon_id'))
    auth_chain = list(block.find_values('_pdbx_poly_seq_scheme.pdb_strand_id'))
    auth_seq_pos = list(block.find_values('_pdbx_poly_seq_scheme.auth_seq_num'))
    ['standard_res', 'standard_chain', 'standard_seq_pos', 'auth_res', 'auth_chain', 'auth_seq_pos']

    data = {
        'standard_res': standard_res,
        'standard_chain': standard_chain,
        'standard_seq_pos': standard_seq_pos,
        'auth_res': auth_res,
        'auth_chain': auth_chain,
        'auth_seq_pos': auth_seq_pos
    }

    cif_seq_data = pd.DataFrame(data)
    
    # Map unique chain names and extract seq data
    unique_cif_seq_data = cif_seq_data[['standard_chain', 'auth_chain']].drop_duplicates()

    chain_map = {}

    for auth, std in zip(unique_cif_seq_data['auth_chain'], unique_cif_seq_data['standard_chain']):
        chain_map[auth] = std
    
    for unique_chain in cif_atom_data['auth_chain'].unique():
        if unique_chain in chain_seq_map:
            chain_id = chain_map[unique_chain]
            seq = chain_seq_map[unique_chain]
            
            # Write to fasta
            outfile = filename.replace('.cif', '_{}.fa'.format(unique_chain))

            with open(outfile, 'w') as out_handle:
                out_handle.write('>{}\n{}\n'.format(chain_id, seq))
            