from concurrent.futures import ThreadPoolExecutor
import docker
import subprocess
import os
import sys
from tqdm import *
import glob
import time
import subprocess

def process(pdb_id):
    
    if not os.path.exists('/home/tshimko/tesselate/data/{}.pdb'.format(pdb_id)):
        pdb_file = '/shared_data/protein_structure/PDB/pdb/pdb{}.ent.gz'.format(pdb_id)
        
        
        with open('/home/tshimko/tesselate/data/{}.log'.format(pdb_id), 'w') as logfile:
            subprocess.call(['python',
                             '/home/tshimko/tesselate/scripts/pdbtools/clean_pdb.py',
                             '-O',
                             '/home/tshimko/tesselate/data/cleaned',
                             '-rmw',
                             pdb_file], stderr=logfile)
    
    client = docker.from_env()
    uid = os.geteuid()
        
    vols = {'/home/tshimko/tesselate/data/cleaned': {'bind': '/data_in', 'mode': 'ro'},
            '/home/tshimko/tesselate/data/processed': {'bind': '/data_out', 'mode': 'rw'}}
    
    cmd_string = 'python /arpeggio/arpeggio.py -v -od /data_out /data_in/{}.pdb'
    cmd = cmd_string.format(pdb_id)
    
    container = client.containers.run('tcs_arpeggio', command=cmd, volumes=vols, user=uid)
    
    return 'PDB ID {} complete.'.format(pdb_id)

if __name__ == '__main__':
    
    with open(sys.argv[1], 'r') as handle:
        pdb_ids = [acc.strip().lower() for acc in handle.readlines()]
    
    pool = ThreadPoolExecutor(35)
    
    futures = []
    
    contacts = glob.glob('data/processed/*.contacts')
    atomtypes = glob.glob('data/processed/*.atomtypes')
    
    for pdb in tqdm(pdb_ids):
        if 'data/processed/{}.contacts'.format(pdb) not in contacts or 'data/processed/{}.atomtypes'.format(pdb) not in atomtypes:
            futures.append(pool.submit(process, (pdb)))
         
    n_complete = [future.done() for future in futures]
    
    while sum(n_complete) != len(futures):
        print(sum(n_complete), 'of', len(futures), 'complete processes.')
        time.sleep(10)
        n_complete = [future.done() for future in futures]
    