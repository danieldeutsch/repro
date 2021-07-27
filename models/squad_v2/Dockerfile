FROM python:3.6

WORKDIR /app

# Install python dependencies
RUN pip install --no-cache-dir numpy==1.19.5

# Download the evaluation script and data file from the official codalab
# https://worksheets.codalab.org/worksheets/0xbe2859a20b9e41d2a2b63ea11bd97740
RUN wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O evaluate-v2.0.py
RUN wget https://worksheets.codalab.org/rest/bundles/0xb30d937a18574073903bb38b382aab03/contents/blob/ -O dev-v2.0.json