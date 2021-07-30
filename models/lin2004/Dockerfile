FROM python:3.6

WORKDIR /app

# Install a necessary Perl library
RUN apt update && apt install -y libxml-dom-perl

# Download the original ROUGE Perl code
RUN pip3 install --no-cache-dir gdown
RUN gdown https://drive.google.com/uc?id=1K4J2wHGjAyr3LoSgaQuWZ_YyjtUGf26m
RUN unzip ROUGE-1.5.5-Linux.zip
RUN rm ROUGE-1.5.5-Linux.zip

# Copy over the file to run the sentence splitting and install
# its dependencies
COPY src/sentence_split.py sentence_split.py
RUN pip install --no-cache-dir nltk==3.6.2
RUN python -c 'import nltk; nltk.download("punkt")'
