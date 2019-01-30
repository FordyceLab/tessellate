import numpy as np
import os
import h5py
import sys

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

if __name__ == '__main__':
    
    h5file = h5py.File('./data/contacts.hdf5', 'w')
    
    atomtypes_group = h5file.create_group('atomtypes')
    memberships_group = h5file.create_group('memberships')
    contacts_group = h5file.create_group('contacts')
    
    with open(sys.argv[1], 'r') as handle:
        accessions = [acc.strip().lower() for acc in handle.readlines()]
        
    for entry in accessions:
        
        atom_list, mem_mat, contacts = process_arpeggio_out('./data', entry)
        
        atomtypes_group.create_dataset(entry, data=atom_list, compression='gzip', compression_opts=9)
        memberships_group.create_dataset(entry, data=mem_mat, compression='gzip', compression_opts=9)
        contacts_group.create_dataset(entry, data=contacts, compression='gzip', compression_opts=9)
        
    h5file.close()
