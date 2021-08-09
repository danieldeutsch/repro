FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained models. The --no-same-owner flag
# is to prevent an error message when tar runs as root.
# See https://superuser.com/questions/1435437/how-to-get-around-this-error-when-untarring-an-archive-tar-cannot-change-owner
RUN wget https://storage.googleapis.com/sfr-factcc-data-research/factcc-checkpoint.tar.gz
RUN tar -xvzf factcc-checkpoint.tar.gz --no-same-owner
RUN rm factcc-checkpoint.tar.gz

RUN wget https://storage.googleapis.com/sfr-factcc-data-research/factccx-checkpoint.tar.gz
RUN tar -xvzf factccx-checkpoint.tar.gz --no-same-owner
RUN rm factccx-checkpoint.tar.gz

# The default version of cryptacular, 1.6.0, which is installed as a dependency of
# a package in requirements.txt, gave us errors installing:
# ERROR: Could not build wheels for cryptacular which use PEP 517 and cannot be installed directly
RUN pip install --no-cache-dir cryptacular==1.5.1

# Clone the code and install the dependencies
RUN git clone https://github.com/salesforce/factCC && \
    cd factCC && \
    git checkout 40a8a5684ec7131bfc92396d9b7f236f80095665 && \
    pip install --no-cache-dir -r requirements.txt

# Install extra dependencies
RUN pip install --no-cache-dir wandb==0.11.2

# Copy the scoring code
COPY src/run_test.py factCC/modeling/run_test.py

# Run warmup code
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh