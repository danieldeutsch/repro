FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained model
RUN wget https://nlp.biu.ac.il/~pyatkiv/roleqsmodels/question_transformation.tar.gz && \
    tar -zxf question_transformation.tar.gz && \
    rm question_transformation.tar.gz

# Install the Python dependencies
# These were specified in the repository's Readme
RUN pip install --no-cache-dir \
    torch==1.7.1 \
    spacy==2.3.2 \
    transformers==4.1.1 \
    allennlp==1.2.0rc1

# There were found by running the code and installing missing packages.
# nltk was set to 3.4.5 because the default installed version made changes
# which broke this code.
RUN pip install --no-cache-dir \
    qanom==0.0.6 \
    jsonlines==2.0.0 \
    pattern==3.6 \
    nltk==3.4.5 \
    unidecode==1.3.2 \
    pytorch_lightning==1.5.4

# Download the code
RUN git clone https://github.com/danieldeutsch/RoleQGeneration && \
    cd RoleQGeneration && \
    git checkout 34a03b2c054bdf0361a0ba3b70482261cb89cf57

RUN python -c 'import nltk; nltk.download("verbnet"); nltk.download("wordnet"); nltk.download("wordnet_ic"); nltk.download("sentiwordnet"); nltk.download("averaged_perceptron_tagger");'