from concurrent.futures import ThreadPoolExecutor
import docker
import subprocess
import os
import sys
from tqdm import *
import glob
import time

def process(pdb_id):
    
    client = docker.from_env()
    uid = os.geteuid()
        
    vols = {'/shared_data/protein_structure/PDB/cif': {'bind': '/data_in', 'mode': 'ro'},
            '/home/tshimko/tesselate/data': {'bind': '/data_out', 'mode': 'rw'}}
    
    cmd_string = 'python /arpeggio/arpeggio.py -v -od /data_out /data_in/{}.cif.gz'
    cmd = cmd_string.format(pdb_id)
    
    container = client.containers.run('tcs_arpeggio', command=cmd, volumes=vols, user=uid)
    
    return 'PDB ID {} complete.'.format(pdb_id)

if __name__ == '__main__':
    
    with open(sys.argv[1], 'r') as handle:
        pdb_ids = [acc.strip().lower() for acc in handle.readlines()]
    
    pool = ThreadPoolExecutor(35)
    
    futures = []
    
    contacts = glob.glob('data/*.contacts')
    atomtypes = glob.glob('data/*.atomtypes')
    
    for pdb in tqdm(pdb_ids):
        if 'data/{}.contacts'.format(pdb) not in contacts or 'data/{}.atomtypes'.format(pdb) not in atomtypes:
            futures.append(pool.submit(process, (pdb)))
         
    n_complete = [future.done() for future in futures]
    
    while sum(n_complete) != len(futures):
        print(sum(n_complete), 'of', len(futures), 'complete processes.')
        time.sleep(10)
        n_complete = [future.done() for future in futures]
    