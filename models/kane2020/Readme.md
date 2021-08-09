# Kane et al. (2020)

## Publication
[NUBIA: NeUral Based Interchangeability Assessor for Text Generation](https://arxiv.org/abs/2004.14667)

## Repositories
https://github.com/wl-research/nubia

## Available Models
- Nubia
  - Description: A learned text generation evaluation metric
  - Name: `kane2020-nubia`
  - Usage: Include a small snippet for how to use the model
    ```python
    from repro.models.kane2020 import NUBIA
    model = NUBIA()
    inputs = [
        {"candidate": "The candidate text", "references": ["The reference text"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` is the Nubia score averaged over the inputs, and `micro` is the Nubia score per-input.
    
## Implementation Notes
- The implementation does not support using a GPU
- The metric only supports a single reference, so the length of `references` must be 1.
    
## Docker Information
- Image name: `danieldeutsch/kane2020:1.0`
- Build command:
  ```shell script
  repro setup kane2020 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup kane2020
pytest models/kane2020/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1114187440).
We replicated the features show in an example from the original repository.
However, there are additional features now and the overall score has changed.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog