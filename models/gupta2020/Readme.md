# Gupta et al. (2020)

## Publication
[Neural Module Networks for Reasoning over Text](https://arxiv.org/abs/1912.04971)

## Repositories
https://github.com/nitishgupta/nmn-drop

## Available Models
- [Model trained on DROP](https://drive.google.com/drive/folders/1N1gCOJPndF2BHOMU-igV3X_SCdizGFbf)
  - Description: A model which uses neural module networks to answer questions on the DROP dataset 
  - Name: `gupta2020-nmn`
  - Usage:
    ```python
    from repro.models.gupta2020 import NeuralModuleNetwork
    model = NeuralModuleNetwork()
    answer = model.predict("context paragraph", "question")
    ```
    
## Implementation Notes
The implementation here was based on the instructions in the [original repository](https://github.com/nitishgupta/nmn-drop).
    
## Docker Information
- Image name: `gupta2020`
- Docker Hub: https://hub.docker.com/repository/docker/danieldeutsch/gupta2020
- Build command:
  ```
  repro setup gupta2020 [--silent]
  ```
- Requires network: No
  
## Testing
```
repro setup gupta2020
pytest models/gupta2020/tests
```

## Status
- [x] Regression unit tests pass  
- [x] Correctness unit tests pass  
The unit tests check to make sure the model returns the same answers for a few examples in their demo.
[Latest successful test on Github.](https://github.com/danieldeutsch/repro/actions/runs/1050729637)
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested
