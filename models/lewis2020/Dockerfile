FROM pure/python:3.7-cuda10.0-base

ARG CNNDM
ARG XSUM

WORKDIR /app

# Download the pretrained models
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Clone fairseq, checkout latest tested commit, and install dependencies
RUN git clone https://github.com/pytorch/fairseq && \
    cd fairseq && \
    git checkout f2146bdc7abf293186de9449bfa2272775e39e1d && \
    pip install --no-cache-dir --editable .

# Install additional libraries that are required
RUN pip install --no-cache-dir requests
