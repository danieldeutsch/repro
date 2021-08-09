# Scialom et al. (2019)

## Publication
[Answers Unite! Unsupervised Metrics for Reinforced Summarization Models](https://arxiv.org/abs/1909.01610)

## Repositories
https://github.com/ThomasScialom/summa-qa

## Available Models
This implementation wraps the `SummaQA` metric.

- SummaQA
  - Description: A reference-free QA-based metric 
  - Name: `scialom2019-summaqa`
  - Usage:
    ```python
    from repro.models.scialom2019 import SummaQA
    model = SummaQA()
    inputs = [
        {"candidate": "The candidate", "references": ["The reference"], "sources": ["The source"]},
        ...
    ]   
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` is the averaged SummaQA scores over the inputs, and `micro` is the individual scores per input.
    
    
## Implementation Notes
- The metric only supports 1 source document, so the length of `sources` must be 1.
- The metric does not support using the GPU
    
## Docker Information
- Image name: `scialom2019`
- Build command:
  ```shell script
  repro setup scialom2019 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup scialom2019
pytest models/scialom2019/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1113804331).
We have reproduced the examples from the original code's Github repo.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested