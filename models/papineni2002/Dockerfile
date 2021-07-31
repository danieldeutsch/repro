FROM python:3.6

WORKDIR /app

# Install SacreBLEU
RUN pip install --no-cache-dir sacrebleu==1.5.1

# Copy scoring scripts
COPY src/sentbleu.py sentbleu.py
COPY src/bleu.py bleu.py
