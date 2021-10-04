# Yuan et al. (2021)

## Publication
[BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/abs/2106.11520)

## Repositories
https://github.com/neulab/BARTScore

## Available Models
- BARTScore
  - Description: A text generation evaluation metric based on BART
  - Name: `yuan2021-bartscore`
  - Usage:
    ```python
    from repro.models.yuan2021 import BARTScore
    model = BARTScore()
    inputs = [
        {"candidate": "The candidate text", "references": ["The references"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` and `micro` are the average and per-input BARTScores.
    You can pass `model="parabank"` to use the Parabank trained model.
    
## Implementation Notes
Currently only the base model `facebook/bart-large-cnn` is supported.
    
## Docker Information
- Image name: `danieldeutsch/yuan2021:1.0`
- Docker Hub:
- Build command:
  ```shell script
  repro setup yuan2021 [--silent]
  ```
- Requires network: Yes, there is still a request sent although the models are pre-cached.
  
## Testing
```shell script
repro setup yuan2021
pytest models/yuan2021/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
We verify the outputs on their Github Readme.
See [here]().
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog