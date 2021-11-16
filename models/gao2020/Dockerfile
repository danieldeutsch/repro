FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Checkout the code from our fork
RUN git clone https://github.com/danieldeutsch/SUPERT && \
    cd SUPERT && \
    git checkout a70d931a08dd5840c7bf8d726174b9dcde31031a

# Install the python dependencies
RUN cd SUPERT && pip install -r requirements.txt --no-cache-dir

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run a warmup query to pre-cache models
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
