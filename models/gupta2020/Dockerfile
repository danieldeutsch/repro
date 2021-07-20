FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained model
RUN pip3 install --no-cache-dir gdown
RUN gdown https://drive.google.com/uc?id=14gYu4cEhz_KMsCASAl7zHu-Jsk0dxsZs

# Setup semparse and checkout the commit specified by https://github.com/nitishgupta/nmn-drop/issues/8#issuecomment-625834039
RUN git clone https://github.com/allenai/allennlp-semparse && \
    cd allennlp-semparse && \
    git checkout 937d594

# Install python dependencies
RUN pip install --no-cache-dir \
    allennlp==0.9 \
    dateparser==0.7.2
RUN python -m spacy download en_core_web_lg

# allennlp==0.9 does not fix the overrides version. There has since been a breaking
# change. Fix to a working version
RUN pip install --no-cache-dir overrides==3.1.0

# Setup the code
RUN git clone https://github.com/nitishgupta/nmn-drop && \
    cd nmn-drop && \
    git checkout 1ff2047be76689e914cca35b9ace713f3e502577 && \
    ln -s ../allennlp-semparse/allennlp_semparse/ ./

# Run a warmup example to pre-cache models
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh