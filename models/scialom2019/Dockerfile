FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# We manually install the code's requirements because some of
# the dependencies are fixed, while others are not, and this has
# caused breaking changes
RUN pip install --no-cache-dir \
    torch==1.1.0 \
    spacy==2.2.0 \
    transformers==2.1.1 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm

# Clone the metric code
RUN git clone https://github.com/recitalAI/summa-qa.git && \
    cd summa-qa && \
    git checkout 39bdaeafc922dbd704bdc4b4af9e587516b831cb && \
    pip install --no-cache-dir .

# Copy the scoring script
COPY src/score.py score.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh