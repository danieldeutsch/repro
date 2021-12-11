FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install the library
RUN pip install --upgrade pip && \
    pip install --no-cache-dir unbabel-comet==1.0.1 --use-feature=2020-resolver

# Run a warmup query
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh