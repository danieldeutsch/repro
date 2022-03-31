# Using Models in Repro
This tutorial will cover some basics about what users of Repro need to know to use the models in the library.

## Predict and Predict Batch
Each model currently has a `predict()` and `predict_batch()` method.
The `predict()` method is meant for running the model on a single example, whereas `predict_batch()` is for multiple, which is often significantly more efficient than calling `predict()` multiple times.
`predict()` is almost always a wrapper around `predict_batch()` which wraps the example in a list.

The input to `predict_batch()` is typically a list of examples, where each example is a dictionary.
The keys for those dictionaries should correspond to the parameters of `predict()`.
Here is an example:
```python
from typing import Dict, List
from repro.models import Model

class MyModel(Model):
    def predict(self, candidate: str, reference: str, **kwargs) -> str:
        return self.predict_batch([{"candidate": candidate, "reference": reference}], **kwargs)[0]
    
    def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[str]:
        candidates = [inp["candidate"] for inp in inputs]
        references = [inp["reference"] for inp in inputs]
        # Run the rest of predict_batch
        pass
```
Optional parameters to `predict()` should be handled accordingly by `predict_batch()`.

## GPU Access
Some models use a GPU.
To restrict a model to use a particular GPU, you have to pass that device ID to the model's `device` parameter.
Setting the `CUDA_VISIBLE_DEVICES` environment variable does not work.

## Downloading the Docker Image
Each model has a Docker Image published on Docker Hub.
The images are versioned to ensure that the code and the Docker images are always compatible.

When you run a model for the first time and do not specific a specific Docker Image, the default image will be downloaded from Docker Hub.
This default image name is defined in `models/<model-name>/__init__.py`.
If you want to directly download this image, you can run
```shell script
repro pull <model-name>
```
Since the image versions are hard-coded, new updates to the image should not break your repro version since they will use updated version tags. 

## Creating a Custom Docker Image
Each image which is published to Docker Hub uses the model's default configuration.
You may need to use a custom image which contains optional pre-trained models, for example, which will require re-building the Docker image.
This can be done via
```shell script
repro setup <model-name> --image-name <image-name>
```
Run `repro setup <model-name> --help` to see what other parameters you can specify.

Then, you have to pass your new image name to your model (the name that you specified for `<image-name>`).
For example, `model = MyModel(image="new-image")`.

## Documentation
Each model has its own Readme under its `models/<model-name>` directory, which should list useful informationi about what models are implemented.
The Repro model's Python wrapper is located under `repro/models/<model-name>` in case you need to understand what its parameters mean.
