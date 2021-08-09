FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the model code
RUN git clone https://github.com/wl-research/nubia.git && \
    cd nubia && \
    git checkout abf15bcb5a3c4a23c192fd53a48bab6b0a1e0cb3 && \
    pip install --no-cache-dir .

# Copy the scoring script
COPY src/score.py score.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh