FROM python:3.6

WORKDIR /app

RUN pip install --no-cache-dir sacrerouge==0.2.3
RUN python -c 'import nltk; nltk.download("punkt")'

# Install ROUGE perl dependencies
RUN apt-get update && \
    apt-get install -y libxml-dom-perl

RUN sacrerouge setup-metric rouge

# Copy over utility files
COPY src/sentence_split.py sentence_split.py