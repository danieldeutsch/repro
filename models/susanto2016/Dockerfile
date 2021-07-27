FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Download the pretrained models. They're small enough
# that we always take all of them
RUN wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/lrl-truecaser-model-ru.tar.gz
RUN wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wiki-truecaser-model-en.tar.gz
RUN wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wmt-truecaser-model-de.tar.gz
RUN wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wmt-truecaser-model-es.tar.gz


# Install the python dependencies. We fix the overrides version because
# there is a breaking change made in a later version
RUN pip install --no-cache-dir \
    allennlp==0.8.2 \
    scikit-learn==0.22.2 \
    overrides==3.1.0

# Clone the repo with the code
RUN git clone https://github.com/mayhewsw/pytorch-truecaser && \
    cd pytorch-truecaser && \
    git checkout 6d42d0f86b95f24db57935d0dae069d97e020fd5
