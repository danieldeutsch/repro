FROM pure/python:3.6-cuda9.0-base

WORKDIR /app

RUN apt-get update && apt-get install curl -y

# Install Python packages. The original code requires only
# torch==0.4.0 and allennlp==0.5.0. The rest of the packages
# were found based on trial and error
RUN pip install \
    https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl \
    allennlp==0.5.0 \
    overrides==3.1.0 \
    numpydoc \
    scikit-learn==0.22.2 \
    networkx

# Download the code
RUN git clone https://github.com/nafitzgerald/nrl-qasrl && \
    cd nrl-qasrl && \
    git checkout 9369e53adb22dcd5ae8310d31334d9f16bcfcdc9

# Download pretrained models
RUN cd nrl-qasrl/scripts && ./download_pretrained.sh

# Download other dependencies
RUN wget https://nlp.stanford.edu/data/glove.6B.zip && \
    unzip glove.6B.zip && \
    mkdir -p nrl-qasrl/data/glove && \
    cat glove.6B.100d.txt | gzip > nrl-qasrl/data/glove/glove.6B.100d.txt.gz && \
    rm glove*

RUN wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 && \
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json && \
    mkdir -p nrl-qasrl/data/elmo && \
    mv elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 nrl-qasrl/data/elmo && \
    mv elmo_2x4096_512_2048cnn_2xhighway_options.json nrl-qasrl/data/elmo

# Run a warmup query to download cached items
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh