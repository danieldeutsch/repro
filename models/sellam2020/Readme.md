# Sellam et al. (2020)

## Publication
[BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696)

## Repositories
https://github.com/google-research/bleurt

## Available Models
The `BLEURT` class can be instantiated with the checkpoints provided by the original repository.
See [here](https://github.com/google-research/bleurt/blob/master/checkpoints.md) for the list.
The corresponding model names are `"BLEURT-20"`, `"BLEURT-20-{D12,D6,D3}"` or `"bleurt-{tiny,base,large}-{128,512}"` and should be passed to the constructor of the class.

- BLEURT
  - Description: A learned evaluation metric for natural language generation
  - Name: `sellam2020-bleurt`
  - Usage:
    ```python
    from repro.models.sellam2020 import BLEURT
    model = BLEURT(model="BLEURT-20")
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
    [--not-bleurt-20] \
    [--tiny-512] \
    [--base-512] \
    [--large-128] \
    [--large-512] \
    [--bleurt-20-d12] \
    [--bleurt-20-d6] \
    [--bleurt-20-d3] \
    [--silent]
  ```
  The arguments specify which BLEURT models should be downloaded.
  `BLEURT-20`, `bleurt-tiny-128`, and `bleurt-base-128` are downloaded by default.
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
See [here](https://github.com/danieldeutsch/repro/actions/runs/2016646079).
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
### v1.1
- Upgraded to set BLEURT-20 as the default model and use the faster length-batched implementation
