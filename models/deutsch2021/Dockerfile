FROM pure/python:3.7-cuda10.0-base

ARG QG
ARG QA
ARG BERTSUMEXTABS_CNNDM
ARG BERTSUMEXTABS_XSUM

WORKDIR /app

# Download the pretrained models
RUN pip3 install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Install the QAEval library
RUN pip install --no-cache-dir qaeval

# Copy over files for running prediction
COPY src/generate_questions.py generate_questions.py
COPY src/answer_questions.py answer_questions.py

# Run warmup queries
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh