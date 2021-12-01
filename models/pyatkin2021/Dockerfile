FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained model
RUN wget https://nlp.biu.ac.il/~pyatkiv/roleqsmodels/question_transformation.tar.gz && \
    tar -zxf question_transformation.tar.gz && \
    rm question_transformation.tar.gz

# Download the code
RUN git clone https://github.com/ValentinaPy/RoleQGeneration && \
    cd RoleQGeneration && \
    git checkout 168eb664d146b7e4e50b56a1ac6ea7a25743e797

# Install the Python dependencies
RUN pip install --no-cache-dir qanom==0.0.3

RUN pip install --no-cache-dir jsonlines
RUN pip install --no-cache-dir pattern

RUN python -c 'import nltk; nltk.download("wordnet"); nltk.download("wordnet_ic"); nltk.download("sentiwordnet"); nltk.download("verbnet")'

RUN pip install --no-cache-dir \
    nltk==3.4.5 \
    spacy \
    unidecode \
    pytorch_lightning