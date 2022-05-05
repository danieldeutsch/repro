FROM danieldeutsch/python:3.8-cuda11.1.1-base

WORKDIR /app

# Download the code
RUN git clone https://github.com/ufal/MTEQA && \
    cd MTEQA && \
    git checkout d1d8b676af4f840ce5a712a41215e880daf81764 && \
    git submodule init && \
    git submodule update

# Install the required libraries. The library requires torch==1.8.0
# but the default CUDA version is incompatible, so we reinstall the right one
RUN cd MTEQA && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Download the models
RUN cd MTEQA && python download_models.py

# Download NLTK models
RUN python -c "import nltk; nltk.download('punkt')"