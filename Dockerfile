#Dockerfile for running cobaya notebooks with binder
#https://mybinder.org/v2/gh/CobayaSampler/cobaya/master?filepath=docs%2Fcobaya-example.ipynb

FROM cmbant/cosmobox

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

WORKDIR ${HOME}

RUN pip install getdist --user

RUN python setup.py install --user
