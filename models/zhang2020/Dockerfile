FROM pure/python:3.7-cuda10.0-base

ARG MODELS

WORKDIR /app

# Install bert_score
RUN pip install --no-cache-dir bert_score==0.3.9

# Copy over the scoring script
COPY src/score.py score.py

# Run the warmup examples, which download the models
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
