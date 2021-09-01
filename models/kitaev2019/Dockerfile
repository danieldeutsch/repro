FROM pure/python:3.7-cuda10.0-base

ARG MODELS

WORKDIR /app

# Install the library
RUN pip install --no-cache-dir \
    benepar==0.2.0 \
    click==7.1.1

# Download the spacy dependency
RUN python -m spacy download en_core_web_md

# Download the pretrained parsers
COPY src/download_models.py download_models.py
RUN python download_models.py --models $MODELS

# Copy the prediction code
COPY src/parse.py parse.py