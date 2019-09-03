FROM continuumio/miniconda3

MAINTAINER Tyler Shimko <tshimko@stanford.edu>

RUN apt-get update && \
    apt-get install -y libnetcdf-dev tcl8.6-dev && \
    apt-get clean && \
    conda install -c conda-forge vmd-python && \
    conda install tk=8.6 numpy scipy expat matplotlib \
    scikit-learn pytest pandas seaborn cython && \
    git clone https://github.com/getcontacts/getcontacts.git
    
ENV PATH "$PATH:/getcontacts"
