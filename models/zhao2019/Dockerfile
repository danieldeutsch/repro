FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install the Python dependencies
RUN git clone https://github.com/AIPHES/emnlp19-moverscore && \
    cd emnlp19-moverscore && \
    git checkout 8ba2bfd10a58bf2a0f19fe1c2cf2edcb16957391 && \
    pip install --no-cache-dir .

# The setup.py does not contain all of the dependenices, so we manually install them.
# We fix specific versions to prevent future breaking changes
RUN pip install --no-cache-dir \
    torch==1.9.0 \
    pyemd==0.5.1 \
    transformers==4.9.0 \
    sentencepiece==0.1.96 \
    matplotlib==3.4.2

# Download the stopwords file
RUN wget https://raw.githubusercontent.com/AIPHES/emnlp19-moverscore/master/examples/stopwords.txt

# Copy the inference code
COPY src/score.py score.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh