# Tessellate deep contact map prediction

This repository contains the code associated with the Tessellate contact map prediction project. The major directories and files are outlined below.

## `Snakefile`

Snakemake pipeline to handle all data processing steps. Some steps require docker containers. These containers can either be built form some of the Dockerfiles found in the containers directory or directly form the [Biocontainers repository](https://biocontainers.pro/). All other scripts for data processing steps can be found in the `scripts` subdirectory of this repository.

## `containers`

Contains a number of Dockerfiles for custom containers necessary for the processing of the raw PDB files. 

## `scripts`

Contains all of the custom necessary to preprocess the raw files from the PDB into sets of covalent bond graphs, atom identities, and non-covalent contacts.

## `tessellate`

Old processining and model components, currently obsolete. This directory may be employed in the future for packaging functional model components or data preprocessing steps.

## `notebooks`

Notebooks used to test individual modules (whether for data processesing/loading or for models) or create and run full models.

### `benchmarks`

Contains all notebooks used to create and train all models. Notebooks found in the `single_example` directory were only trained on the `6E6O` structure without a validation dataset. Noetbooks in the `multiple_example` directory were trained on the list of structures outlined in the Chapter 5 document document found in this repository.

### `module_sandbox`

Contains sandboxed (devlopment versions) of data processing and model modules. This directory is generally used for testing purposes only.
