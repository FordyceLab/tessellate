FROM continuumio/miniconda3

MAINTAINER Tyler Shimko <tshimko@stanford.edu>

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    conda install -c schrodinger pymol
