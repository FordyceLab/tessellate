import pandas as pd
import re
from numpy import nan
import sys

if __name__ == '__main__':
    
    # Read command line args
    comp_number = sys.argv[1]
    out_file = sys.argv[2]
    
    # Get the table
    tables = pd.read_html('http://predictioncenter.org/casp{}/targetlist.cgi'.format(comp_number),
                          attrs={'class': 'table'})

    # Sort inverted by length
    tables.sort(key=len, reverse=True)
    
    # Get rid of poorly handled header
    pdb_lookup = tables[0].dropna(subset=[0]).iloc[2:, 1:].dropna(axis=1).iloc[:, [0, -1]]

    # Make pattern to extract PDB IDs
    pdb_pattern = re.compile('[1-9][0-9|a-z]{3}')
    
    # Make a list of PDB IDs
    pdb_ids = []

    # Loop through the table and extract the PDB IDs
    for i in pdb_lookup.iloc[:, 1]:
        pdb_field = i.split()[-1]
        match = pdb_pattern.match(pdb_field)
        if match is not None:
            pdb_ids.append(match.string)
        else:
            pdb_ids.append(nan)

    # Remake the table as a lookup
    pdb_lookup.iloc[:, 1] = pdb_ids
    pdb_lookup.iloc[:, 0] = [i.split()[0] for i in pdb_lookup.iloc[:, 0]]
    pdb_lookup.dropna(inplace=True)
    
    # Export to tsv
    pdb_lookup.to_csv(out_file, sep='\t', header=False, index=False)
    