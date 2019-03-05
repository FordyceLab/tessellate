PN_PARENT_DIR  = '/shared_data/protein_structure'
LOCAL_DATA_DIR = 'data'
ID_LIST_DIR = 'id_lists'

ALL_IDS = ['training_{}'.format(thres) for thres in [30, 50, 70, 90, 95, 100]] + ['validation', 'testing']
TYPES = ['complete', 'x_ray', 'nmr', 'cryo_em']
COMPS = [11, 12]
TYPE_KEY_DICT = {'x_ray': 'diffraction', 'nmr': 'NMR', 'cryo_em': 'EM'}


# Process all complete IDs for a given competition
rule process_PN_ID_lists:
    input:
        expand(ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/{type}/{dataset}_ids.txt',
               comp_number = COMPS, type=TYPES, dataset=ALL_IDS)
    

# Clean up the id list directory
rule clean:
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
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/training_{threshold,\d+}_ids.txt'
    shell:
        'cat {input}/training_{wildcards.threshold} | awk "/\[ID\]/{{getline; print}}" | sed "s/_/\t/g" | cut -f1 | tr "[:upper:]" "[:lower:]" > {output}'
        

# Process the ProteinNet validation ID data sets
rule process_PN_validation_set:
    input:
        PN_PARENT_DIR + '/ProteinNet_CASP{comp_number}'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/validation_ids.txt'
    shell:
        'cat {input}/validation | awk "/\[ID\]/{{getline; print}}" | sed "s/#/\t/g" | sed "s/_/\t/g" | cut -f2 | tr "[:upper:]" "[:lower:]" > {output}'


# Process the ProteinNet testing ID data sets
rule process_PN_testing_set:
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/complete/testing_ids.txt'
    shell:
        'python scripts/convert_targets_to_pdb.py {wildcards.comp_number} {output}'
        
rule filter_type_structures:
    input:
        type_file = ID_LIST_DIR + '/pdb_info/{type}_structures.txt',
        ids = ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number}/complete/{in_file}.txt'
    output:
        ID_LIST_DIR + '/ProteinNet/ProteinNet{comp_number,\d+}/{type}/{in_file}.txt'
    shell:
        'comm -12 <( sort {input.type_file} ) <( sort {input.ids} ) > {output}'


# Download PDB entry type list
rule download_PDB_entry_type:
    output:
        ID_LIST_DIR + '/pdb_info/pdb_entry_type.txt'
    shell:
        'wget -P {ID_LIST_DIR}/pdb_info ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt'
     

rule parse_structure_type:
    input:
        ID_LIST_DIR + '/pdb_info/pdb_entry_type.txt'
    output:
        ID_LIST_DIR + '/pdb_info/{type}_structures.txt'
    params:
        type_key = lambda wildcards: TYPE_KEY_DICT[wildcards.type]
    shell:
        'grep {params.type_key} {input} | cut -f1 > {output}'
