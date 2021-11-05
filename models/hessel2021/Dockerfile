FROM danieldeutsch/python:3.7-cuda11.0.3-base

WORKDIR /app

# Install the CLIPScore library. The torch version was set based on the CLIP
# library Readme. The corresponding torchvision version was selected accordingly.
# This version of numpy was required. See https://github.com/jmhessel/clipscore/issues/2
# sklearn is not automatically installed but required.
RUN git clone https://github.com/jmhessel/clipscore && \
    cd clipscore && \
    git checkout 74b5871ac1feb9f31b304a820c464b183a5cabcd && \
    pip install torch==1.7.1 torchvision==0.8.2 numpy==1.20.3 sklearn && \
    pip install -r requirements.txt

# Copy over an image necessary for a warmup query, then run a warmup
RUN mkdir images
COPY tests/fixtures/image1.jpeg images/image1.jpeg
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
