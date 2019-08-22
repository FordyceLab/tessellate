#############
# Variables #
#############

PN_PARENT_DIR  = '/shared_data/protein_structure'
LOCAL_DATA_DIR = 'data'
ID_LIST_DIR = 'id_lists'

ALL_IDS = ['training_{}'.format(thres) for thres in [30, 50, 70, 90, 95, 100]] + ['validation', 'testing']
TYPES = ['x_ray', 'nmr', 'cryo_em']
COMPS = [11, 12]
TYPE_KEY_DICT = {'x_ray': 'diffraction', 'nmr': 'NMR', 'cryo_em': 'EM'}

if 'list' in config:
    PROC_TARGETS = [acc.strip() for acc in open(config['list'], 'r').readlines()]
else:
    PROC_TARGETS = []
    
shell.executable('bash')

###############################
# Clean and process PDB files #
###############################

rule process_PDB_list:
    input:
        expand(LOCAL_DATA_DIR + '/processed/{pdb_id}.atomtypes',
               pdb_id=PROC_TARGETS),
        expand(LOCAL_DATA_DIR + '/processed/{pdb_id}.contacts',
               pdb_id=PROC_TARGETS)
               
rule process_PDB_file:
    input:
        LOCAL_DATA_DIR + '/cleaned/{pdb_id}.pdb'
    output:
        LOCAL_DATA_DIR + '/processed/{pdb_id}.atomtypes',
        LOCAL_DATA_DIR + '/processed/{pdb_id}.contacts'
    shell:
        'docker run -v $(pwd)/{LOCAL_DATA_DIR}/cleaned:/data_in \
            -v $(pwd)/{LOCAL_DATA_DIR}/processed:/data_out \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            tcs_arpeggio:latest \
            python /arpeggio/arpeggio.py -v -od /data_out /data_in/{wildcards.pdb_id}.pdb'
            
rule clean_PDB_file:
    input:
        PN_PARENT_DIR + '/PDB/pdb/pdb{pdb_id}.ent.gz'
    output:
        LOCAL_DATA_DIR + '/cleaned/{pdb_id}.pdb'
    shell:
        'python scripts/pdbtools/clean_pdb.py\
            -O {LOCAL_DATA_DIR}/cleaned -rmw {input}'


###############################
# Process ProteinNet ID lists #
###############################

# Process all complete IDs for a given competition
rule process_PN_ID_lists:
    input:
        expand(ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/{type}/{dataset}_ids.txt',
               comp_number=COMPS, type=TYPES, dataset=ALL_IDS),
        expand(ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/x_ray/success/{dataset}_ids.txt',
               comp_number=COMPS, dataset=ALL_IDS)
    

# Clean up the id list directory
rule clean_ID_lists:
    shell:
        'rm -r {ID_LIST_DIR}'
    

# Download the ProteinNet dataset for a desired competition
rule download_ProteinNet:
    output:
        directory(PN_PARENT_DIR + '/ProteinNet_CASP{comp_number}')
    shell:
        'wget -O - "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp{wildcards.comp_number}.tar.gz" | tar xzf - -C /shared_data/protein_structure/ProteinNet_CASP{wildcards.comp_number} --strip-components 1'
        
        
# Process the ProteinNet training ID data sets
rule process_PN_training_sets:
    input:
        PN_PARENT_DIR + '/ProteinNet_CASP{comp_number}'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/all/training_{threshold,\d+}_ids.txt'
    shell:
        'cat {input}/training_{wildcards.threshold} | awk "/\[ID\]/{{getline; print}}" | sed "s/_/\t/g" | cut -f1 | tr "[:upper:]" "[:lower:]" > {output}'
        

# Process the ProteinNet validation ID data sets
rule process_PN_validation_set:
    input:
        PN_PARENT_DIR + '/ProteinNet_CASP{comp_number}'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/all/validation_ids.txt'
    shell:
        'cat {input}/validation | awk "/\[ID\]/{{getline; print}}" | sed "s/#/\t/g" | sed "s/_/\t/g" | cut -f2 | tr "[:upper:]" "[:lower:]" > {output}'


# Process the ProteinNet testing ID data sets
rule process_PN_testing_set:
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/all/testing_ids.txt'
    shell:
        'python scripts/convert_targets_to_pdb.py {wildcards.comp_number} {output}'

# Get current structures
rule get_current_sturcture_list:
    input:
        PN_PARENT_DIR + '/PDB/pdb'
    output:
        ID_LIST_DIR + '/pdb_info/all_current_structures.txt'
    shell:
        'find {input} -type f -printf "%f\n" | sed "s/^pdb//g" | sed "s/\.ent\.gz//g" | sort > {output}'

# Filter all lists to only have current structures
rule filter_current_structures:
    input:
        curr_file = ID_LIST_DIR + '/pdb_info/all_current_structures.txt',
        ids = ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/complete/all/{in_file}.txt'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/complete/current/{in_file}.txt'
    shell:
        'comm -12 <( cat {input.curr_file} ) <( sort {input.ids} ) > {output}'

# Filter structures by type
rule filter_type_structures:
    input:
        type_file = ID_LIST_DIR + '/pdb_info/{type}_structures.txt',
        ids = ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/complete/current/{in_file}.txt'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/{type([a-z|\_])+}/{in_file}.txt'
    shell:
        'comm -12 <( sort {input.type_file} ) <( sort {input.ids} ) > {output}'


# Download PDB entry type list
rule download_PDB_entry_type:
    output:
        ID_LIST_DIR + '/pdb_info/pdb_entry_type.txt'
    shell:
        'wget -P {ID_LIST_DIR}/pdb_info ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt'
     
# Parse the structure type for each entry
rule parse_structure_type:
    input:
        ID_LIST_DIR + '/pdb_info/pdb_entry_type.txt'
    output:
        ID_LIST_DIR + '/pdb_info/{type}_structures.txt'
    params:
        type_key = lambda wildcards: TYPE_KEY_DICT[wildcards.type]
    shell:
        'grep {params.type_key} {input} | cut -f1 > {output}'
        
# Subset existing list by data sets that were succesfully processed
rule parse_successful_examples:
    input:
        data_file = LOCAL_DATA_DIR + '/contacts.hdf5',
        list_file = ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/{type}/{in_file}.txt'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/{type,([a-z|\_])+}/success/{in_file}.txt'
    shell:
        'python scripts/filter_ID_list_success.py {input.data_file} {input.list_file} > {output}'


#######################
# Sync local PDB copy #
#######################

rule sync_PDB:
    input:
        PN_PARENT_DIR + '/PDB/pdb'
    shell:
        'bash scripts/rsyncPDB.sh'
