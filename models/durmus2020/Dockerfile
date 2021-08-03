FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /app

RUN apt update && apt install -y git

# Cython and numpy need to be installed before benepar, one of FEQA's dependencies
RUN pip install --no-cache-dir \
    Cython==0.29.15 \
    numpy==1.19.1

# Download the code and install dependencies
RUN git clone https://github.com/esdurmus/feqa && \
    cd feqa && \
    git checkout 60a996970b005d1321c86a7ad10327bfd7e197b6 && \
    pip install --no-cache-dir -r requirements.txt

# Download the pretrained models
RUN pip install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Install extra dependencies. The example code shows en_core_web_sm==2.1.0, but we ran into an exception
# while running that version:
#
#   ValueError: [E167] Unknown morphological feature: 'ConjType' (9141427322507498425). This can happen
#   if the tagger was trained with a different set of morphological features. If you're using a pretrained model,
#   make sure that your models are up to date
#
# After changing to 2.3.1, this error went away
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz#egg=en_core_web_sm
RUN python -c 'import benepar; benepar.download("benepar_en2")'
RUN python -c 'import nltk; nltk.download("stopwords"); nltk.download("punkt")'

# Copy the scoring code
COPY src/score.py feqa/score.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh