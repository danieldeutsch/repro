FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install the SacreROUGE and QAEval libraries
RUN pip install --no-cache-dir \
    sacrerouge==0.2.3 \
    qaeval==0.0.9 \
    click==7.1.2 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm

# Setup the QAEval metric
RUN sacrerouge setup-metric qa-eval

# Copy over files for running prediction
COPY src/generate_questions.py generate_questions.py
COPY src/answer_questions.py answer_questions.py

# Run warmup queries
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh