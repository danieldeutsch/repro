# Rei et al. (2020)

## Publication
[COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/)

## Repositories
https://github.com/Unbabel/COMET

## Available Models
The available models are COMET using the reference-based `wmt20-comet-da` model or the reference-free `wmt20-comet-qe-da` model. 

- COMET:
  - Description: A machine translation evaluation metric.
  - Name: `rei2020-comet`
  - Usage:
    ```python
    from repro.models.rei2020 import COMET
    model = COMET()
    # reference-based
    inputs = [
        {"candidate": "The candidate to score", "sources": ["The source text"], "reference": ["The reference"]}
    ]
    macro, micro = model.predict_batch(inputs)
    
    # reference-free
    inputs = [
        {"candidate": "The candidate to score", "sources": ["The source text"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` and `micro` are the averaged and input-level COMET scores.
    The reference-based key is `"comet"` and the reference-free key is `"comet-src"`.
    
## Implementation Notes
Only 1 source document and 1 reference translation are supported.
    
## Docker Information
- Image name: `danieldeutsch/rei2020:1.0`
- Build command:
  ```shell script
  repro setup rei2020
  ```
- Requires network: Yes, the code still makes a network request even if the models are pre-cached.
  
## Testing
```shell script
repro setup rei2020
pytest models/rei2020/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1567865901) 
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Changelog
