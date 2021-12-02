FROM pure/python:3.7-cuda10.0-base

ARG MODELS

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ZhangShiyue/Lite2-3Pyramid && \
    cd Lite2-3Pyramid && \
    git checkout 6d15d64e3dce5eee584cead5836ecf6a556b3d42

## Install the python dependencies. The Github repo requires torch==1.7.1+cu110, but
## we use torch==1.7.1 with cuda 10.0 instead
RUN pip install --no-cache-dir \
    transformers==4.9.2 \
    torch==1.7.1 \
    tqdm==4.62.3 \
    nltk==3.6.3 \
    allennlp==2.6.0 \
    allennlp_models==2.6.0 \
    click==7.0

# Install nltk dependencies
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('wordnet_ic'); nltk.download('sentiwordnet')"

# Run the warmup code
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh