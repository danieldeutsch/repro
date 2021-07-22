FROM pure/python:3.7-cuda10.0-base

ARG TRANSFORMERABS_CNNDM
ARG BERTSUMEXT_CNNDM
ARG BERTSUMEXTABS_CNNDM
ARG BERTSUMEXTABS_XSUM

WORKDIR /app

# Install OpenJDK-8, with a workaround via https://linuxize.com/post/install-java-on-debian-10/
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common && \
    wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && \
    add-apt-repository -y https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ && \
    apt update && \
    apt install -y adoptopenjdk-8-hotspot

# Download CoreNLP, required for preprocessing
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
RUN unzip stanford-corenlp-full-2017-06-09.zip
RUN rm stanford-corenlp-full-2017-06-09.zip
ENV CLASSPATH=stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar

# Download the pretrained models
RUN pip3 install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Clone the code, switch to the latest tested commit "25c24abd236f4aa896660e1e6dc509d4f3aec372"
# on the "dev" branch with the code that can process raw text
RUN git clone https://github.com/nlpyang/PreSumm
RUN cd PreSumm && git checkout 25c24abd236f4aa896660e1e6dc509d4f3aec372

# Setup the python environment. We can't directly pip install the requirements from PreSumm
# because torch==1.1.0 cannot be directly installed with pip anymore. It has to be installed via the whl file.
# First install torch, then install the other dependencies
RUN pip3 install --no-cache-dir wheel
RUN wget https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN rm torch-1.1.0-cp37-cp37m-linux_x86_64.whl

# Install other dependencies based on the requirements.txt
RUN pip3 install --no-cache-dir \
    multiprocess==0.70.9 \
    numpy==1.17.2 \
    pyrouge==0.1.3 \
    pytorch-transformers==1.2.0 \
    tensorboardX==1.9

# Copy over the script to preprocess the data
COPY src/preprocess.py preprocess.py

# Run some code which loads the BERT model so it already available in the cache.
# This is an optimization.
COPY src/warmup.py warmup.py
RUN python3 warmup.py
