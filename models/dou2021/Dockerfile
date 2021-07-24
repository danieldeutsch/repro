FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install OpenJDK-8, with a workaround via https://linuxize.com/post/install-java-on-debian-10/
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common && \
    wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && \
    add-apt-repository -y https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ && \
    apt update && \
    apt install -y adoptopenjdk-8-hotspot

# Download CoreNLP, required for preprocessing
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
RUN unzip stanford-corenlp-full-2016-10-31.zip
RUN rm stanford-corenlp-full-2016-10-31.zip
ENV CLASSPATH=stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Download the pretrained models
RUN pip install --no-cache-dir gdown
COPY scripts/download-models.sh download-models.sh
RUN sh download-models.sh

# Clone the code, switch to the latest tested commit
RUN git clone https://github.com/neulab/guided_summarization
RUN cd guided_summarization && git checkout ea4bbe91f189cdb51f7f6a827210f9adc5319b3c

# These are the dependencies taken from the original fairseq repo, not the
# one under "guided_summarization/bart/fairseq", which does not have a setup.py
# or requirements.txt. We also fix torch==1.4.0, which is the minimum version
# stated by the authors. We ran into errors with torch==1.9.0
RUN pip install --no-cache-dir \
    cffi \
    cython \
    dataclasses \
    hydra-core==1.0 \
    omegaconf==2.1 \
    numpy \
    regex \
    sacrebleu==1.4.12 \
    torch==1.4.0 \
    tqdm

# Install spacy and en_core_web_sm, required for our sentence tokenization and
# extracting the oracle keywords
RUN pip install --no-cache-dir \
    spacy==3.1.1 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz

# Move the models into the "guided_summarization" directory
RUN mv bart_sentence guided_summarization/bart_sentence

# Download dependencies for the preprocessing
RUN cd guided_summarization/bart && \
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' && \
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' && \
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

# Copy over the code to run prediction and tokenization
COPY src/summarize.py guided_summarization/bart/summarize.py
COPY src/sentence_tokenize.py sentence_tokenize.py
