FROM tensorflow/tensorflow:2.2.2-gpu

WORKDIR /app

# Download utils
RUN apt update && apt install -y wget git

# Download the pretrained models
RUN pip3 install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh

ARG TINY_128
ARG TINY_512
ARG BASE_128
ARG BASE_512
ARG LARGE_128
ARG LARGE_512
ARG BLEURT_20
ARG BLEURT_20_D12
ARG BLEURT_20_D6
ARG BLEURT_20_D3

RUN sh download-models.sh

# Install BLEURT dependencies. Their setup.py does not fix versions,
# which resulted in installing a newer tensorflow version which
# was incompatible with the version of cuda installed by the tensorflow/tensorflow:2.2.2-gpu
# image. Therefore, we directly install the dependencies with specific versions
RUN pip install --no-cache-dir \
    tensorflow==2.2.2 \
    pandas==1.1.5 \
    scipy==1.5.4 \
    tf-slim==1.1.0 \
    sentencepiece==0.1.96

# Install BLEURT
RUN git clone https://github.com/google-research/bleurt && \
    cd bleurt && \
    git checkout c6f2375c7c178e1480840cf27cb9e2af851394f9 && \
    pip install --no-cache-dir .
