FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the code and checkout v0.1.1, which has the latest working
# version for summarization according to the Readme
RUN git clone https://github.com/ThomasScialom/QuestEval && \
    cd QuestEval && \
    git checkout 7c827804a8da82560e91cf8fe84124d37f7c0660

# Install torch==1.9.0, which is not fixed in the metric code
RUN pip install --no-cache-dir torch==1.9.0

# Install the Python dependencies
RUN cd QuestEval && \
    pip install --no-cache-dir .

# Install the unilm dependencies
RUN cd QuestEval/unilm/s2s-ft && \
    pip install --no-cache-dir .

# The default `datasets` has an error that `pyarrow>=1.0.0` needs to
# be installed. We instead fix the `datasets` to 1.7.0, which is the
# latest version of the library in the QuestEval repro
RUN pip install --no-cache-dir datasets==1.7.0

# Copy over the inference code
COPY src/score.py score.py

# Run a warmup query
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
