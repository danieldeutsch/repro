# Zhao et al. (2019)

## Publication
[MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://aclanthology.org/D19-1053/)

## Repositories
https://github.com/AIPHES/emnlp19-moverscore

## Available Models
This implementation wraps `MoverScore` from the original repository.
The `MoverScore` model leaves all of the optional parameters as default.
The `MoverScoreForSummarization` uses a stopword file following their example code.

- MoverScore
  - Description: A text generation evaluation metric
  - Name: `zhao2019-moverscore`
  - Usage:
    ```python
    from repro.models.zhao2019 import MoverScore
    model = MoverScore()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` results are the MoverScore scores averaged over the `inputs`.
    The `micro` results are the MoverScore results for each item in `inputs`.
    
- MoverScoreForSummarization
  - Description: A variant of `MoverScore` which uses stopwords by default based on the example code.
  - Name: `zhao2019-moverscore-summarization`
  - Usage:
    ```python
    from repro.models.zhao2019 import MoverScoreForSummarization
    model = MoverScoreForSummarization()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` results are the MoverScore scores averaged over the `inputs`.
    The `micro` results are the MoverScore results for each item in `inputs`.
    
## Implementation Notes
- The current `moverscore` Python code does not appear to support GPUs, so even if you pass a GPU device to `device`, it will still run on CPU. 
    
## Docker Information
- Image name: `zhao2019`
- Build command:
  ```shell script
  repro setup zhao2019 [--silent]
  ```
- Requires network: Yes, the library queries for a file even if the file is locally in the cache.
  
## Testing
```shell script
repro setup zhao2019
pytest models/zhao2019/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1091879442)
- [ ] Correctness unit tests pass  
None provided in the original repo.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested