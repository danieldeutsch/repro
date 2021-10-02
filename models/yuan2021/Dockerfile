FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained Parabank model
RUN pip install --no-cache-dir gdown && \
    gdown https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m

# Download the code and install the dependencies
RUN git clone https://github.com/neulab/BARTScore && \
    cd BARTScore && \
    git checkout a93654e9f28f6997dc2cc5106a41f7a3ad0db91a && \
    pip install --no-cache-dir -r requirements.txt

# Copy over the scoring code
COPY src/score.py score.py