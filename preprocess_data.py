import multiprocessing as mp
from tesselate.data import process
import h5py
from tqdm import *
import numpy as np
import time

def write_hdf5(result):
    written = False
    
    while not written:
        try:
            with h5py.File('data/training.hdf5', 'a') as out_dataset:
                entry_group = out_dataset.create_group(result['id'][0])
                entry_group.create_dataset('atomtypes', data=result['atomtypes'][0], compression='gzip', compression_opts=3)
                entry_group.create_dataset('memberships', data=result['memberships'][0], compression='gzip', compression_opts=3)
                entry_group.create_dataset('target', data=result['target'][0], compression='gzip', compression_opts=3)
                entry_group.create_dataset('atom_adjacency', data=result['atom_adjacency'][0], compression='gzip', compression_opts=3)
                entry_group.create_dataset('res_adjacency', data=result['res_adjacency'][0], compression='gzip', compression_opts=3)
                entry_group.create_dataset('combos', data=result['combos'][0], compression='gzip', compression_opts=3)

            written = True

        except OSError:
            time.sleep(0.5)
            

def prep(entry):
    
    if 'dataset' not in locals():
        dataset = h5py.File('data/contacts.hdf5', 'r')
    
#     tqdm.write('Proc {}: rec {}'.format(name, entry))
    
#     with h5py.File('data/contacts.hdf5', 'r') as dataset:

    # Read the data from the HDF5 file
    atomtypes = dataset[entry]['atomtypes'][:].astype(np.int64)
    memberships = dataset[entry]['memberships'][:]
    contacts = dataset[entry]['contacts'][:]

    # Handle the target slightly differently
    # Rearrange columns
    target = dataset[entry]['target'][:][:, [2, 0, 1]]
        
#     tqdm.write('Proc {}: read {}'.format(name, entry))

    data_dict = process(entry, atomtypes, contacts, memberships, target, 'complete')
    
#     tqdm.write('Proc {}: ret {}'.format(name, entry))
    
    write_hdf5(data_dict)
    

if __name__ == '__main__':
    
    # Init the pool
    pool = mp.Pool(38)
    
    
    with h5py.File('data/contacts.hdf5', 'r') as dataset:
        keys = list(dataset.keys())
    
    results = pool.imap(prep, keys)
    
    for result in tqdm(results, dynamic_ncols=True, total=len(keys)):
        pass
        
    pool.terminate()
    pool.join()
