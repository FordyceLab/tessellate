#############
# Variables #
#############

PN_PARENT_DIR  = '/shared_data/protein_structure'
LOCAL_DATA_DIR = 'data'
ID_LIST_DIR = 'id_lists'
TMP_DIR = 'tmp'

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
            
            
################
# Get contacts #
################

checkpoint split_cif_file:
    input:
        PN_PARENT_DIR + '/PDB/cif/{pdb_id}.cif.gz'
    output:
        TMP_DIR + '/{pdb_id}/{pdb_id}-{number,\d+}.cif'
    shell:
        'mkdir -p {TMP_DIR}/{wildcards.pdb_id} && \
        gunzip -c {input} > {TMP_DIR}/{wildcards.pdb_id}/{wildcards.pdb_id}.cif && \
        docker run -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}:/data \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            assemblies:latest \
            Assemblies.py {wildcards.pdb_id}.cif'
        
rule mv_bioassemblies:
    input:
        checkpoints.split_cif_file.get().output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}.cif'
    shell:
        'mv {input} {TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}/'
        
rule add_h:
    input:
        rules.mv_bioassemblies.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_hydro.pdb'
    shell:
        'docker run --rm -i \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}:/data pymol:latest \
            pymol -c -d "set cif_use_auth, off; \
                load /data/{wildcards.pdb_id}-{wildcards.number}.cif; \
                h_add; \
                save /data/{wildcards.pdb_id}-{wildcards.number}_hydro.pdb"'
                
rule clean_h:
    input:
        rules.add_h.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_hydro.mmcif'
    shell:
        'cat {input} | \
            awk \'/_atom_site.auth_asym_id/ {{print "_atom_site.auth_seq_id\\n_atom_site.auth_comp_id" }}1\' | \
            awk \'/_atom_site.pdbx_PDB_model_num/ {{print \"_atom_site.auth_atom_id\"}}1\' | \
            awk \'{{OFS = \"\\t\"; if ($1 ==  \"ATOM\" || $1 == \"HETATOM\") $17=$9 FS $6 FS $17 FS $4; print}}\' > {output}'

rule get_full_contacts:
    input:
        rules.add_h.output
    output:
        LOCAL_DATA_DIR + '/contacts/{pdb_id}-{number,\d+}.full_contacts'
    shell:
        'docker run --rm -it \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/tmp/{wildcards.pdb_id}/{wildcards.number}:/data_in \
            -v $(pwd)/{LOCAL_DATA_DIR}/contacts:/data_out \
            getcontacts:latest \
            get_static_contacts.py --structure /data_in/{wildcards.pdb_id}-{wildcards.number}_hydro.pdb \
                --output /data_out/{wildcards.pdb_id}-{wildcards.number}.full_contacts --itypes all'
                
rule simplify_contacts:
    input:
        rules.get_full_contacts.output
    output:
        LOCAL_DATA_DIR + '/contacts/{pdb_id}-{number,\d+}.contacts'
    shell:
        'grep -v \'#\' {LOCAL_DATA_DIR}/contacts/{wildcards.pdb_id}-{wildcards.number}.full_contacts | \
                awk \'BEGIN{{OFS=\"\\t\"}} {{print $2,$3,$4}}\' > {output}'

                
##############
# Get graphs #
##############

# Handle the structure files directly

rule remove_solvent:
    input:
        rules.mv_bioassemblies.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_no_solvent.cif'
    shell:
        'docker run --rm -i \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}:/data pymol:latest \
            pymol -c -d "set cif_use_auth, off; \
                load /data/{wildcards.pdb_id}-{wildcards.number}.cif; \
                remove (hydro); remove solvent; \
                save /data/{wildcards.pdb_id}-{wildcards.number}_no_solvent.cif"'
                
rule cif_no_solvent2molreport:
    input:
        rules.remove_solvent.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_no_solvent.molreport'
    shell:
        'docker run --rm -i \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}:/data \
            informaticsmatters/obabel:latest \
                obabel /data/{wildcards.pdb_id}-{wildcards.number}_no_solvent.cif \
                -O /data/{wildcards.pdb_id}-{wildcards.number}_no_solvent.molreport'
                
rule cif_no_solvent2id:
    input:
        rules.remove_solvent.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_no_solvent.id'
    shell:
        'awk \'/^(HET)?ATO?M/{{OFS=\"\\t\"; print $2,$7":"$6":"$9":"$4,$3}}\' \
         {input} > {output}'
                

# Extract the sequences and get full graph of just amino acid sequences

rule cif_extract_fasta:
    input:
        rules.mv_bioassemblies.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_{chain,\S}.fa'
    shell:
        'python scripts/extract_polymer_seqs_from_cif.py {input}'
            
rule fasta2molreport:
    input:
        rules.cif_extract_fasta.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_{chain,\S}.molreport'
    shell:
        'docker run --rm -i \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}:/data \
            informaticsmatters/obabel:latest \
                obabel /data/{wildcards.pdb_id}-{wildcards.number}_{wildcards.chain}.fa \
                -d \
                -O /data/{wildcards.pdb_id}-{wildcards.number}_{wildcards.chain}.molreport'

rule fasta2id:
    input:
        rules.cif_extract_fasta.output
    output:
        TMP_DIR + '/{pdb_id}/{number,\d+}/{pdb_id}-{number}_{chain,\S}.id'
    shell:
        'docker run --rm -it \
            -u $(id -u ${{USER}}):$(id -g ${{USER}}) \
            -v $(pwd)/{TMP_DIR}/{wildcards.pdb_id}/{wildcards.number}:/data \
            informaticsmatters/obabel:latest \
                obabel \
                /data/{wildcards.pdb_id}-{wildcards.number}_{wildcards.chain}.fa \
                -opdb | \
                awk \'/^COMPND/{{CHAIN = $2}} /^(HET)?ATO?M/{{OFS=\"\\t\"; $5 = CHAIN; \
                    print $2,$5":"$4":"$6":"$3,$12}}\' \
                > {output}'
                
rule process_graph:
    input:
        rules.fasta2molreport.output,
        rules.fasta2id.output,
        rules.cif_no_solvent2molreport.output,
        rules.cif_no_solvent2id.output
    output:
        nodes = LOCAL_DATA_DIR + '/graphs/{pdb_id}-{number,\d+}_nodes.csv',
        edges = LOCAL_DATA_DIR + '/graphs/{pdb_id}-{number}_edges.csv',
        mask = LOCAL_DATA_DIR + '/graphs/{pdb_id}-{number}_mask.txt'
    shell:
        'python scripts/combine_ids_and_molreport.py {output.nodes} {output.edges} {output.mask} {input}'
        

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
        PN_PARENT_DIR + '/PDB/pdb',
        PN_PARENT_DIR + '/PDB/cif'
    shell:
        'bash scripts/rsyncPDB.sh'
