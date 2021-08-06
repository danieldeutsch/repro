# Name of Model

## Publication
List the relevant publications for this model

## Repositories
List the relevant repositories for this model

## Available Models
List the models available and their usage.
For example:

- [Model name](link to pretrained model)
  - Description: A description of what this model is, what it corresponds to in the relevant publication, etc.
  - Name: The name of the model as registered in `repro`
  - Usage: Include a small snippet for how to use the model
    ```python
    from repro.models.example import Example
    model = Example()
    model.predict()
    ```
    
## Implementation Notes
Dicuss anything relevant to someone who wants to use this model.
    
## Docker Information
- Image name: List the name of the corresponding docker image
- Docker Hub: Add the link to the image on Docker Hub if it exists
- Build command: Provide documentation on how to build the image
  ```shell script
  repro setup <model-name> \
    --arg1 \
    [--arg2]
  ```
  Provide an explanation for the arguments
- Requires network: Yes/No, indicate whether running the image requires Internet access.
  Ideally, it should not in order to ensure the container will be able to run even if its dependencies no longer exist online.
  
## Testing
Explain how to run the unittests for this model
```shell script
repro setup <model-name> \
    --arg1 \
    --arg2
pytest models/<model-name>/tests
```

## Status
Mark the current status of this implementation, providing details as necessary (e.g., pointers to relevant shell scripts)

- [ ] Regression unit tests pass  
This means that the unit tests check for some particular output for a given input, although it might not be the "correct" output. 
- [ ] Correctness unit tests pass  
This means that the unit tests check for some particular output for a given input, and that output is what is considered "correct," for example by reproducing an example input/output pair given by the original repository.
- [ ] Model runs on full test dataset  
This means the model can successfully make predictions on the entire test dataset without crashing
- [ ] Predictions approximately replicate results reported in the paper  
This means that even if the predictions don't get the exact scores reported in the paper or original repository, they are close enough to consider this a successful reimplementation
- [ ] Predictions exactly replicate results reported in the paper  
This means that the predictions and results are exactly what is reported in the paper or original repository.

## Changelog
List the changes for each new version