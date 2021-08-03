FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained models
RUN pip install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Install the QAEval library
RUN pip install --no-cache-dir \
    qaeval==0.1.0 \
    click==7.1.2 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm

# Copy over files for running prediction
COPY src/generate_questions.py generate_questions.py
COPY src/answer_questions.py answer_questions.py
COPY src/score.py score.py

# Run warmup queries
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh