FROM ubuntu:18.04

COPY ucsf-chimerax_0.9_amd64.deb /

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install /ucsf-chimerax_0.9_amd64.deb && \
    rm /ucsf-chimerax_0.9_amd64.deb
    