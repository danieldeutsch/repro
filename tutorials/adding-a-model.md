# Adding a New Model
This tutorial will teach you the steps required to add a new model to Repro.

1. **Create a new directory for your model in the `models` folder.**

    The name should be indicative of the relevant publication or project that it is wrapping (e.g., `deutsch2020` for wrapping models from Deutsch et al. (2020)).

    The contents of this directory should be the following:
    
    - `Readme.md`: This will contain the documentation for your model.
      Start by copying `models/Template.md` into your Readme file.
    - `Dockerfile`: This is the Dockerfile which will be used to build the image for your model.
    - `src/`: This directory will contain the Python source code for your model as well as any Python files used in the Docker image.
    - `src/model.py` or `src/models.py`: This should contain your class which interacts with the Docker container.
    - `src/setup.py`: This will contain the code to add a `repro setup` command for your image.
    - `tests/`: This will contain the unit tests to test the model implementation.
    - `scripts/`: This will contain any shell scripts you include in the Docker container, for example, ones to download model files or run warmup examples.
    - `experiments/`: This directory will contain any experiments that you run to verify a model reproduces results reported in its respective paper.
    
2. **Setup the plumbing for building your Docker container.**

    The "hello-world" Dockerfile for Repro would look something like this:
    ```Dockerfile
    FROM pure/python:3.7-cuda10.0-base
    WORKDIR /app
    ```
    Which creates an image with Python 3.7 and access to the host's GPUs.
    
    Then, create a `src/setup.py` that will build your Docker image:
    ```python
    from repro import MODELS_ROOT
    from repro.commands.subcommand import SetupSubcommand
    from repro.common.docker import BuildDockerImageSubcommand


    @SetupSubcommand.register("my-model")
    class MyModelSetupSubcommand(BuildDockerImageSubcommand):
        def __init__(self) -> None:
            super().__init__("my-model", f"{MODELS_ROOT}/my-model")
    ```
    In this tutorial, our model will be called `"my-model"`, but you should put your model's name (e.g., `"deutsch2020"`) and change the name of the class accordingly.
    
    Next, you have to add an import of class in Repro so it is discovered when building the command line tools.
    To do that, add a file `repro/models/my_model.py` with the following (yes, it is weird):
    ```python
    from ._models.my_model.src.setup import MyModelSetupSubcommand
    ```
    
    Once this is done, you should be able to run
    ```shell script
    repro setup my-model
    ```
    and the Docker image should be built.
    You can verify with
    ```shell script
    docker image ls
    ```
    
3. **Setup the plumbing for running your model**

    Create a file `src/model.py` (or `src/models.py`), which will contain the wrapper of your model:
    ```python
    from typing import Dict, List
    
    from repro.models import Model


    @Model.register("my-model")
    class MyModel(Model):
        def __init__(
            self, image: str = "my-model", device: int = 0
        ) -> None:
            self.image = image
            self.device = device
            
        def predict(text: str, **kwargs) -> str:
            return self.predict_batch([{"text": text}], **kwargs)[0]
            
        def predict_batch(inputs: List[Dict[str, str]], **kwargs) -> List[str]:
            return [inp[::-1] for inp in inputs]
    ```

    
The `model.py` and `setup.py` should be as lightweight as possible.
Any heavy dependencies or preprocessing should be done within the Docker container.
As a general rule, implementing these files should not add new dependencies for Repro.
