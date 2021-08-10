# Adding a Model
This tutorial will cover how to (1) add a new model to the Repro package, (2) test your implementation locally on your machine, and (3) automatically publish your model's Docker image to Docker Hub.

The model name in this tutorial will be called `tutorial`.
When you implement an actual model, your should instead use a name based on the paper that published the model (e.g., `gupta2020`, `deutsch2021`).

# Setup the Dockerfile Plumbing
The first step is to set up the plumbing for building a simple Dockerfile through `repro`.

1. Create a new directory under `models` called `tutorial`.
   This directory will contain your model's Dockerfile, any code or scripts which are added to the Dockerfile, as well as the unit tests for your model.

2. Create a file `models/tutorial/Dockerfile` with the following contents:
   ```Dockerfile
   FROM python:3.7
   WORKDIR /app
   ```
   This Dockerfile will be a Linux distribution with Python 3.7 installed by default.
   If you need GPU support, we recommend starting with base images `pure/python:3.7-cuda10.0-base` if you are using PyTorch and `tensorflow/tensorflow:2.2.2-gpu` for TensorFlow (the exact version number could change based on the TensorFlow version you require).

3. Create a directory `repro/models/tutorial`.
   This directory will contain the wrapper around your model's Docker container, commands to setup/build your model, and any metadata associated with the model.

4. Create a file `repro/models/tutorial/__init__.py` with the following contents:
   ```python
   import os

   VERSION = "1.0"
   MODEL_NAME = os.path.basename(os.path.dirname(__file__))
   DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
   DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
   AUTOMATICALLY_PUBLISH = False
   ```
   This defines some metadata for your model automatically.
   `MODEL_NAME` will be equal to `tutorial` (the name of the `__init__.py` file's directory), `DOCKERHUB_REPRO` points to the Docker Hub repository where your model's image will eventually be published, and `DEFAULT_IMAGE` specifies the name of that Docker image.
   `danieldeutsch` should remain the same (it is the Docker Hub account name we are using for now).
   `AUTOMATICALLY_PUBLISH` indicates whether your Docker image should be published to Docker Hub when it is merged into the master branch.
   For now, we leave it as `False`.

 5. Create a file `repro/model/tutorial/setup.py` with the following contents:
    ```python
    from repro import MODELS_ROOT
    from repro.commands.subcommand import SetupSubcommand
    from repro.common.docker import BuildDockerImageSubcommand
    from repro.models.tutorial import DEFAULT_IMAGE, MODEL_NAME


    @SetupSubcommand.register(MODEL_NAME)
    class TutorialSetupSubcommand(BuildDockerImageSubcommand):
        def __init__(self) -> None:
            super().__init__(f"{MODELS_ROOT}/{MODEL_NAME}", DEFAULT_IMAGE)
    ```
    This class tells `repro` where your model's Dockerfile is located as well as its default image name.

6. Import `TutorialSetupSubcommand` in `__init__.py` **after** the metadata definitions (if you do it before, there will be a circular dependency created):
   ```python
   # repro/model/tutorial/__init__.py
   from repro.models.tutorial.setup import TutorialSetupSubcommand
   ```

7. Now you should be able to build your Dockerfile through `repro`.
   Running `repro setup tutorial` should build the Dockerfile and create an image called `danieldeutsch/tutorial:1.0`.
   You can launch a Docker container with that image and an interactive shell by running
   ```shell script
   docker run -it danieldeutsch/tutorial:1.0 /bin/bash
   ```

## Setup the Model Wrapper
Next, we will create a model which passes data to a Docker container, runs some code in the container, and returns data from it.

1. Create a file `models/tutorial/src/run.py` (and its corresponding directory) with the following contents:
    ```python
    import argparse
    import json


    def main(args):
        with open(args.output_file, "w") as out:
            with open(args.input_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    text = data["text"]
                    reversed_text = text[::-1]
                    out.write(json.dumps({"output": reversed_text}) + "\n")


    if __name__ == '__main__':
        argp = argparse.ArgumentParser()
        argp.add_argument("--input-file", required=True)
        argp.add_argument("--output-file", required=True)
        args = argp.parse_args()
        main(args)
   ```
   This file will be the entrypoint into the Docker container and will actually run the model's code.
   The host is going to create the `--input-file` and the container will read it and write the results to `--output-file`.
   All this code does is reverse the input string, however, your own model implementation may load pre-trained models, run tokenization, etc.
   
2. Add `run.py` to the Docker image by appending this to the `Dockerfile`
   ```Dockerfile
   COPY src/run.py run.py
   ```
   Now when you build your dockerfile, `run.py` will be included.
   Try building the image, running a container, and looking for the file with `ls`.

3. Create a file `repro/models/tutorial/model.py`, with the following contents:
    ```python
    import json
    from typing import Dict, List

    from repro.common.docker import DockerContainer
    from repro.common.io import read_jsonl_file
    from repro.models import Model
    from repro.models.tutorial import DEFAULT_IMAGE, MODEL_NAME


    @Model.register(f"{MODEL_NAME}-model")
    class TutorialModel(Model):
        def __init__(self, image: str = DEFAULT_IMAGE):
            self.image = image
            
        def predict(self, text: str, **kwargs) -> str:
            return self.predict_batch([{"text": text}], **kwargs)[0]
            
        def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[str]:
            with DockerContainer(self.image) as backend:
                host_input_file = f"{backend.host_dir}/input.jsonl"
                host_output_file = f"{backend.host_dir}/output.jsonl"
                container_input_file = f"{backend.container_dir}/input.jsonl"
                container_output_file = f"{backend.container_dir}/output.jsonl"
                
                # Serialize the data to a file
                with open(host_input_file, "w") as out:
                    for inp in inputs:
                        out.write(json.dumps(inp) + "\n")
                        
                # Define the command which will be run in the Docker container
                command = (
                    f"python run.py"
                    f"  --input-file {container_input_file}"
                    f"  --output-file {container_output_file}"
                )
                
                # Run the command in the Docker container
                backend.run_command(command=command)
        
                # Load the results and return
                results = read_jsonl_file(host_output_file)
                raw_results = [result["output"] for result in results]
                return raw_results
   ```
   This file contains the definition of your model that users will import through `repro`.
   It takes the input data from the user, pass it through the Docker container, and return the result.
   Here are some details about what is going on.
   
     - The `@Model.register(f"{MODEL_NAME}-model")` registers your model's implementation so that it can be used through `repro`'s command line API.
       Ours will be named `tutorial-model`.

     - In order to pass data back and forth between the host machine and the Docker container, we mount a directory on the host to the container.
       Any files which are written to the host directory can be read by the Docker container, and any files written by the container to the container's directory can be read by the host.
       However, the directories will have different names, so we keep track of the paths of the parallel files `host_input_file`, `container_input_file`, etc.
       
     - The input data is written to the `host_input_file`.
       It is accessible to the container at `container_input_file`.
       
     - The command which will be run inside of the container is defined.
       The command runs the file which we just created, `run.py`.
       Notice that we use the container's paths and not the host's paths because the command runs inside of the container.
       If you need to run multiple commands, we recommend joining them together with `" && "`.
       
     - The command is run through `backend.run_command()`

     - We read the results from `host_output_file` and return them.
     
   This code should be as lightweight as possible.
   If there are any pre-processing dependencies (e.g., sentence splitting or tokenization), they should be taken care of by `run.py` instead of the model code which will be included in Repro.

4. Import your model in `__init__.py` (again, after the metadata definitions)
   ```python
   from repro.models.tutorial.model import TutorialModel
   ```
   
5. Test to make sure that it works by building the Docker image (`repro setup tutorial`), then instantiating the class
   ```
   >>> from repro.models.tutorial import TutorialModel
   >>> model = TutorialModel()
   >>> model.predict("abc")
   'cba'
   ```
   
## Add Unit Tests
Now that the Dockerfile can be built and the model is implemented, we next add unit tests to ensure that your implementation is correct.

1. The unit tests are contained under `models/tutorial/tests`.
   Create files `models/tutorial/tests/__init__.py` and `models/tutorial/tests/model_test.py`.
   In `model_test.py`, add the following content:
   ```python
   import unittest

   from repro.models.tutorial import TutorialModel
    
    
   class TestTutorialModel(unittest.TestCase):
       def test_tutorial(self):
           model = TutorialModel()
           assert model.predict("abc") == "cba"
           assert model.predict_batch([{"text": "abc"}, {"text": "123"}]) == ["cba", "321"]
   ```
   Then you can run the test from the root of `repro` with `pytest -s models/tutorial/tests`.
   At a minimum, you should add some regression tests (i.e., use your model to predict on a handful of inputs and ensure the output is what you expect).
   This will help to identify if anything changes that might break your implementation.
   If the original code contains its own unit tests or examples, it is good to include those to verify your implementation is faithful to the original.
   
   By default, your model's unit tests will not be run during our regular unit testing.
   Typically these tests are slow, may require GPUs, and take up a lot of resources which the default GitHub unit testing framework cannot support.
   We do have the ability to run them in one-off jobs, but that requires manually triggering the workflow and can only be done by owners of the Repro repository.
   
   If you require data to be loaded (e.g., input examples), they should be placed in a directory called `models/tutorial/tests/fixtures`.
   This directory can be referenced by `models/tutorial/tests/__init__.py`:
   ```python
   import os

   FIXTURES_ROOT = f"{os.path.dirname(os.path.abspath(__file__))}/fixtures"
   ```
   and `FIXTURES_ROOT` can be imported by your tests. 
   
## Add Documentation
Copy the contents of `models/Template.md` to `models/tutorial/Readme.md` and fill out its contents.
Refer to other models' Readmes to see how they have been written.

## Publishing to Docker Hub
Once your implementation has been tested, is working, and approved to be merged, you can change the `AUTOMATICALLY_PUBLISH` variable to `True`.
When your code then gets merged into the master branch, your Docker image will be built and published to `DEFAULT_IMAGE` on [Docker Hub](https://hub.docker.com/u/danieldeutsch/).

## Next Steps
The tutorial model is basic and not very interesting, so the actual implementation of your model may be different.
Here are some tips and reference implementations that you can use when developing your model.

### Adding GPU Access
If you require using a GPU, we recommend using base images `pure/python:3.7-cuda10.0-base` if you are using PyTorch and `tensorflow/tensorflow:2.2.2-gpu` for TensorFlow (if both, default to TensorFlow).
You can change the Python/CUDA/TensorFlow versions as necessary based on what you need.

You should add a `device` parameter to your model's constructor.
We recommend defaulting to 0 (i.e., `device: int = 0`).
If the value is `-1`, that should mean use the CPU.

When you build the command that will run in the container, add `f"export CUDA_VISIBLE_DEVICES={self.device}"` before the code to run the model.
Then, when you load your model in the Docker container, it should be loaded onto GPU ID 0 regardless of the value of `self.device`.

To tell Docker to enable GPU access, you must pass `cuda=True` to `backend.run_command()`.
Otherwise, no GPUs will be visible.

We recommend always being able to run your model with both CPU and GPU.
By default, the unit tests should use CPU (unless it cannot run without GPU).
To enable this, use the following test template:
```python
from parameterized import parameterized
from repro.testing import get_testing_device_parameters

# In your unit test class:
@parameterized.expand(get_testing_device_parameters())
def test_model(self, device: int):
    model = MyModel(device=device)
```
This will run `test_model` once for every GPU or CPU set by the environment variable `TEST_DEVICES`.
If the variable is not set, it will default to only `-1` (the CPU).

### Downloading Pre-Trained Models
It is important to download as many dependences as possible when you build your Docker image.
If you download them when the model initializes, it will do it every time you run `predict()`, which will be very slow.

If the model is accessible via `wget`, it is easiest to directly download the model.
In your Dockerfile:
```Dockerfile
RUN wget http://example.com/model.tar.gz
```

If the model is on Google Drive, you can download it with `gdown`:
```Dockerfile
RUN pip install --no-cache-dir gdown
RUN gdown https://drive.google.com/uc?id=<file-id> --output <file-name>
```

If you have to download a lot of models and do complex setup, we recommend moving that logic into a script called `download-models.sh` under `models/<model-name>/scripts`, then run that script in the Dockerfile.

Finally, if it's difficult to directly download the model and the best way to gather all of the dependencies is to actually load and run the model, we recommend running a warmup example.
In a script called `models/<model-name>/scripts/warmup.sh`, create fake input and run your model's prediction code directly (i.e., the command that you build in your Repro model class).
Copy `warmup.sh` to your Docker image and run it.
This will download any files which are downloaded automatically during initialization (e.g., like in the `transformers` library).
The pre-trained models will be included in the Docker image and won't be downloaded when you run `predict()`.

If your model does not require network access, we recommend passing `network_disabled=True` to `backend.run_command()` to disable the network.
Not requiring network access is ideal because it ensures that your model will always be able to run even if its dependencies are no longer available (as long as the Docker image binaries still exist).

### Example Reference Implementations
Here are some existing implementations of models you can use for reference:

- PyTorch with GPU access: `deutsch2021`, `gupta2020`, `chen2020`, and many others
- TensorFlow with GPU access: `sellam2020`, `durmus2020`
- Downloading pre-trained models with `wget`: `chen2020`
- Downloading pre-trained models `gdown`: `deutsch2021`, and many others
- Running warmup examples: `deutsch2021`, `durmus2020`, `scialom2021`, and many others
- Unit testing with data dependencies: `deutsch2021`, and many others
- Unit testing with CPU and GPUs: `deutsch2021`, and many others
- Unit testing with GPUs only: `scialom2021`
- Configuring which pre-trained models are downloaded based on command line arguments: `liu2019`
