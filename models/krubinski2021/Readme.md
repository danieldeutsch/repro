# Krubinski et al. (2021)

## Publication
[Just Ask! Evaluating Machine Translation by Asking and Answering Questions](https://www.statmt.org/wmt21/pdf/2021.wmt-1.58.pdf)

## Repositories
https://github.com/ufal/MTEQA

## Available Models
- MTEQA:
  - Description: A QA-based evaluation metric for MT
  - Name: `krubinski2021-mteqa`
  - Usage:
    ```python
    from repro.models.krubinski2021 import MTEQA
    model = MTEQA()
    macro, micro = model.predict_batch(
        {"candidate": "The candidate translation", "references": ["The references"]}
    )
    ```
    `macro` and `micro` are the average and per-input scores.
    
## Implementation Notes
    
## Docker Information
- Image name: `danieldeutsch/krubinski2021`
- Docker Hub: 
- Build command:
  ```shell script
  repro setup krubinski2021
  ```
- Requires network: No
  
## Testing
```shell script
repro setup krubinski2021
pytest models/krubinski2021/tests
```

## Status
- [x] Regression unit tests pass   
- [ ] Correctness unit tests pass  
None provided or tested
- [ ] Model runs on full test dataset  
Not Tested
- [ ] Predictions approximately replicate results reported in the paper  
Not Tested
- [ ] Predictions exactly replicate results reported in the paper  
Not Tested

## Changelog
