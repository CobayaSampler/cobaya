"""
.. module:: containers

:Synopsis: Functions and scripts to manage container images
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import logging
from io import StringIO

# Local
from cobaya.log import logger_setup, HandledException
from cobaya.input import get_modules, load_input
from cobaya.yaml_custom import yaml_dump

logger_setup()
log = logging.getLogger(__name__)


base_recipe = ur"""
# OS -------------------------------------------------------------------------
FROM ubuntu:xenial
# POST -----------------------------------------------------------------------
RUN echo "Booting up container -- installing dependencies"
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade
RUN \
  apt-get install -y \
    autoconf automake make \
    gcc-6-base \
    libopenblas-base liblapack3 liblapack-dev libcfitsio-dev \
    python python-pip \
    git wget
# COBAYA  --------------------------------------------------------------------
# getdist fork (it will be an automatic requisite in the future)
RUN pip install pip --upgrade
RUN pip install git+https://github.com/JesusTorrado/getdist/\#egg=getdist
RUN pip install matplotlib # necessary for getdist, but not installed automatically yet
RUN pip install cobaya
ENV CONTAINED=TRUE
# FOR TESTS ------------------------------------------------------------------
RUN pip install pytest-xdist
# Prepare tree for modules ---------------------------------------------------
ENV COBAYA_MODULES /modules
RUN mkdir $COBAYA_MODULES
ENV COBAYA_PRODUCTS /products
RUN mkdir $COBAYA_PRODUCTS
# Compatibility with singularity ---------------------------------------------
RUN ldconfig
RUN echo "Base image created."
"""

MPI_recipe = {
    "docker": u"""
    # MPI -- NERSC: must be MPICH installed in user space
    # http://www.nersc.gov/users/software/using-shifter-and-docker/using-shifter-at-nersc/
    RUN cd /tmp && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
      && tar xvzf mpich-3.2.tar.gz && cd /tmp/mpich-3.2 \
      && ./configure && make -j4 && make install && make clean && rm /tmp/mpich-3.2.tar.gz
    """,
    "singularity": u"""
    RUN apt-get install openmpi-bin
    """}


def get_docker_client():
    try:
        import docker
    except ImportError:
        log.error("The Python Docker interface not installed: do 'pip install docker'.")
        raise HandledException
    return docker.from_env(version="auto")


def create_base_image():
    log.info("Creating base image...")
    dc = get_docker_client()
    stream = StringIO(base_recipe)
    dc.images.build(fileobj=stream, tag="cobaya_base")
    stream.close()
    log.info("Base image created!")


def create_docker_image(filenames):
    log.info("Creating Docker image...")
    dc = get_docker_client()
    modules = yaml_dump(get_modules(*[load_input(f) for f in filenames]))
    echos = "\n".join(['RUN echo "%s" >> /tmp/modules.yaml'%s
                       for s in modules.split("\n")])
    recipe = ur"""
    FROM cobaya_base
    %s
    %s
    RUN pip install cobaya
    
    RUN cobaya-install /tmp/modules.yaml --path /modules --just-code
    """ % (MPI_recipe["docker"], echos)
    stream = StringIO(recipe)
    dc.images.build(fileobj=stream)
    stream.close()
    log.info("Docker image created!")


# Command-line script
def create_image_script():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description=("Cobaya's tool for preparing Docker (for Shifter) "
                     "and Singularity images."))
    parser.add_argument("files", action="store", nargs="+", metavar="input_file.yaml",
                        help="One or more input files.")
    group_type = parser.add_mutually_exclusive_group(required=True)
    group_type.add_argument("-d", "--docker", action="store_const", const="docker",
                            help="Create a Docker image (for Shifter).", dest="type")
    group_type.add_argument("-s", "--singularity", action="store_const", dest="type",
                            const="singularity", help="Create a Singularity image.")
    arguments = parser.parse_args()
    if arguments.type == "docker":
        create_docker_image(arguments.files)
    elif arguments.type == "singularity":
        create_singularity_image(*arguments.files)
