# Tessellate deep contact map prediction

This repository contains the code associated with the Tessellate contact map prediction project. The major directories and files are outlined below.

## `Snakefile`

Snakemake pipeline to handle all data processing steps. Some steps require docker containers. These containers can either be built form some of the Dockerfiles found in the containers directory or directly form the [Biocontainers repository](https://biocontainers.pro/). All other scripts for data processing steps can be found in the `scripts` subdirectory of this repository.

