FROM danieldeutsch/python:3.7-cuda11.0.3-base

WORKDIR /app

# Get the code
RUN git clone https://github.com/PierreColombo/nlg_eval_via_simi_measures && \
    cd nlg_eval_via_simi_measures && \
    git checkout 44d33f34833193a34407cbe6086418c1374ebce6

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.21.5 \
    tqdm==4.62.3 \
    POT==0.8.1.0 \
    torch==1.10.2+cu113 \
    transformers==4.16.1 \
    sklearn \
    geomloss==0.2.4 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Copy over a new CLI file which has an option for an output file
COPY src/score_cli.py nlg_eval_via_simi_measures/score_cli.py

# Run warmup queries
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
