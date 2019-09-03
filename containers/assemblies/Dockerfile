FROM python:3.6

RUN apt-get update && \
    apt-get -y upgrade && \
    pip install pdbx pdbx-mmcif && \
    wget http://mmcif.wwpdb.org/docs/sw-examples/python/src/Assemblies.py && \
    sed -i '1s/^/#\! \/usr\/local\/bin\/python\n\n/' Assemblies.py && \
    chmod +x Assemblies.py && \
    mv Assemblies.py /usr/local/bin
    
VOLUME ["/data"]

WORKDIR /data
    