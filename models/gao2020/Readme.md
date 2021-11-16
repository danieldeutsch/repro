# Gao et al. (2020)

## Publication
[SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization](https://arxiv.org/abs/2005.03724)

## Repositories
Our implementation uses [this fork](https://github.com/danieldeutsch/SUPERT) of the [original repository](https://github.com/yg211/acl20-ref-free-eval)

## Available Models
- SUPERT
  - Description: A reference-free evaluation metric for multi-document summarization
  - Name: `gao2020-supert`
  - Usage:
    ```python
    from repro.models.gao2020 import SUPERT
    model = SUPERT()
    inputs = [
        {"sources": ["The first document", "The second"], "candidate": "The summary to score"}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` and `micro` are the averaged and per-input SUPERT scores.
    
## Implementation Notes
    
## Docker Information
- Image name: `danieldeutsch/gao2020`
- Build command:
  ```shell script
  repro setup gao2020
  ```
- Requires network: No
  
## Testing
```shell script
repro setup gao2020
pytest models/gao2020/tests
```

## Status
- [x] Regression unit tests pass   
See [here](https://github.com/danieldeutsch/repro/actions/runs/1467093505)
- [ ] Correctness unit tests pass  
No expected outputs given in the original repository
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
