# SacreROUGE

## Publication
https://danieldeutsch.github.io/papers/nlp-oss2020/sacrerouge.pdf

## Repositories
https://github.com/danieldeutsch/sacrerouge

## Available Models
SacreROUGE contains implementations of various summarization evaluation metrics.
Thus far, we have added wrappers around ROUGE.

- ROUGE
  - Description: A wrapper around the original Perl implementation of ROUGE in the SacreROUGE library
  - Name: `sacrerouge-rouge`
  - Usage:
    ```python
    from repro.models.sacrerouge import SRROUGE
    model = SRROUGE()
    scores = model.predict("summary", ["reference1", "reference2"])
    ```
    
## Implementation Notes
- If the input summaries/references are strings, the metric will run sentence splitting, which is required to faithfully calculate the ROUGE-L score.
    
## Docker Information
- Image name: `sacrerouge`
- Build command:
  ```shell script
  repro setup sacrerouge [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup sacrerouge
pytest models/<model-name>/tests
```

## Status
- [x] Regression unit tests pass  
- [ ] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1063555327).
We did not check for the correctness of the ROUGE computation.
- [ ] Model runs on full test dataset  
n/a
- [ ] Predictions approximately replicate results reported in the paper  
n/a
- [ ] Predictions exactly replicate results reported in the paper  
n/a