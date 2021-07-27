FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the ptrained model
RUN wget https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz

# Clone the github repo
RUN git clone https://github.com/anthonywchen/MOCHA && \
    cd MOCHA && \
    git checkout debf4d27888d8d36fd5c6d3a9fa02b450e6fbd93

# Install python requirements
RUN cd MOCHA && \
    pip install --no-cache-dir -r requirements.txt

# Copy the inference code
COPY src/predict.py MOCHA/predict.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh