FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install the library
RUN pip install blanc==0.2.2 --no-cache-dir

# Install dependencies
RUN python -c 'import nltk; nltk.download("punkt")'

# Copy over the wrapper code to interact with the library
COPY src/score.py score.py

# Run a warmup query
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh