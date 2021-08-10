FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained model
RUN wget http://data.statmt.org/prism/m39v1.tar && \
    tar -xvf m39v1.tar && \
    rm m39v1.tar

# Clone the model code and install Python dependencies
RUN git clone https://github.com/thompsonb/prism && \
    cd prism && \
    git checkout 42e45a46d1c7924e98bceeed2ea81b31efcb6f9d && \
    pip install --no-cache-dir -r requirements.txt

# Copy the scoring code
COPY src/score.py prism/score.py