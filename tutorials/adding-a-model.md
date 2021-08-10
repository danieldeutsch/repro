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
   FROM pure/python:3.7-cuda10.0-base
   WORKDIR /app
   ```
   This Dockerfile will have Python 3.7 installed by default and include the Nvidia drivers with Cuda 10.0.

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
                    reversed = text[::-1]
                    out.write(json.dumps({"output": reversed}) + "\n")


    if __name__ == '__main__':
        argp = argparse.ArgumentParser()
        argp.add_argument("--input-file", required=True)
        argp.add_argument("--output-file", required=True)
        args = argp.parse_args()
        main(args)
   ```
   This file will be the entrypoint into the Docker container and will actually run the model's code.
   The host is going to create the `--input-file` and the container will read it and write the results to `--output-file`.
   All this code does is reverse the input string.
   
2. Add `run.py` to the Docker image by appending this to the `Dockerfile`
   ```Dockerfile
   COPY src/run.py run.py
   ```
   Now when you build your dockerfile, `run.py` will be included.
   Try building the image, running a container, and looking for the file with `ls`.

3. Create a file `repro/models/tutorial/model.py`, with the following contents:
    ```python
    import json
    from typing import Any, Dict, List, Tuple, Union

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
       
     - The command is run through `backend.run_command()`

     - We read the results from `host_output_file` and return them.

## Example Reference Implementations
Here are some existing implementations of models you can use for reference:

- `chen2020`: Downloads a pre-trained model using `wget
- `sellam2020`: Uses TensorFlow with the GPU
   
