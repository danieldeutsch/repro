# Sellam et al. (2020)

## Publication
[BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696)

## Repositories
https://github.com/google-research/bleurt

## Available Models
The `BLEURT` class can be instantiated with the checkpoints provided by the original repository.
See [here](https://github.com/google-research/bleurt/blob/master/checkpoints.md) for the list.
The corresponding model names are `"bleurt-{tiny,base,large}-{128,512}"` and should be passed to the constructor of the class.

- BLEURT
  - Description: A learned evaluation metric for natural language generation
  - Name: `sellam2020-bleurt`
  - Usage:
    ```python
    from repro.models.sellam2020 import BLEURT
    model = BLEURT(model="bleurt-base-128")
    inputs = [
        {"candidate": "The candidate text", "references": ["The reference", "The other reference"]}
    ]
    scores = model.predict_batch(inputs)
    ```
    
## Implementation Notes
- The original BLEURT code only supports single references.
Our implementation return both the mean and the max BLEURT score over the references (they will be equal if there is only 1 reference).
    
## Docker Information
- Image name: `sellam2020`
- Build command:
  ```shell script
  repro setup sellam2020 \
    [--not-tiny-128] \
    [--not-base-128] \
    [--tiny-512] \
    [--base-512] \
    [--large-128] \
    [--large-512] \
    [--silent]
  ```
  The arguments specify which BLEURT models should be downloaded.
  Both `bleurt-tiny-128` and `bleurt-base-128` are downloaded by default.
- Requires network: No
  
## Testing
Explain how to run the unittests for this model
```shell script
repro setup sellam2020
pytest models/sellam2020/tests
```

## Status
- [x] Regression unit tests pass  
- [x] Correctness unit tests pass  
The unit tests are based on examples in the official repository.
See [here](https://github.com/danieldeutsch/repro/actions/runs/1102526377).
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested