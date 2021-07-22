FROM pure/python:3.7-cuda10.0-base

WORKDIR /app

# Install gsutil to download the model files
# See https://stackoverflow.com/questions/28372328/how-to-install-the-google-cloud-sdk-in-a-docker-image
RUN wget https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -O /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Download the pretrained recipe generation model
RUN gsutil cp gs://roft_saved_models/gpt2-xl_recipes.tar.gz gpt2-xl_recipes.tar.gz
RUN tar -xvf gpt2-xl_recipes.tar.gz
RUN mv finetuned gpt2-xl
RUN rm gpt2-xl_recipes.tar.gz

# Install python dependencies
RUN pip install --no-cache-dir \
    torch==1.9.0 \
    transformers==4.9.0

# Copy over the inference code
COPY src/generate_recipes.py generate_recipes.py

# Run a warmup example
COPY scripts/warmup.sh warmup.sh
RUN sh warmup.sh
