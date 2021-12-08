FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# First setup Prism as a metric
# Download the pretrained model
RUN wget http://data.statmt.org/prism/m39v1.tar && \
    tar -xvf m39v1.tar && \
    rm m39v1.tar

# Clone the model code and install Python dependencies
# We found it necessary to fix the numpy version. At some point, the default
# version that was installed failed because python >=3.8 was required.
RUN pip install --no-cache-dir numpy==1.21.1

RUN git clone https://github.com/thompsonb/prism && \
    cd prism && \
    git checkout 42e45a46d1c7924e98bceeed2ea81b31efcb6f9d && \
    pip install --no-cache-dir -r requirements.txt

# Copy the scoring code
COPY src/score.py prism/score.py

# Now setup Prism as a translation model
# Install sentencepiece build dependencies
RUN apt-get update && apt-get install cmake build-essential pkg-config libgoogle-perftools-dev -y

# Clone the code. The commit ID corresponds to version 0.1.86 which is used by the Prism code
RUN git clone https://github.com/google/sentencepiece && \
    cd sentencepiece && \
    git checkout e8a84a16d13e8bf92892a1cd92e4de3b0d0321fd

# Build the binaries
RUN cd sentencepiece && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig -v

# Copy the translation scripts
COPY src/get_translations.py get_translations.py
COPY scripts/translate.sh translate.sh