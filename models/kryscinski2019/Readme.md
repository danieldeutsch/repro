# Kryściński et al. (2019)

## Publication
[Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)

## Repositories
https://github.com/salesforce/factCC

## Available Models
This implementation wraps the FactCC and FactCCX models.
Both models will return a score and a label.
The score is the probability of the returned label (for binary classification) with label 1 meaning "incorrect".

- FactCC:
  - Description: A model to score the factual consistency of text
  - Name: `kryscinski2019-factcc`
  - Usage:
    ```python
    from repro.models.kryscinski2019 import FactCC
    model = FactCC()
    inputs = [
        {"candidate": "The candidate text", "sources": ["The source text"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` contains the scores averaged over the inputs, whereas `micro` contains the scores for each input.

- FactCCX:
  - Description: A model to score the factual consistency of text
  - Name: `kryscinski2019-factccx`
  - Usage:
    ```python
    from repro.models.kryscinski2019 import FactCCX
    model = FactCCX()
    inputs = [
        {"candidate": "The candidate text", "sources": ["The source text"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` contains the scores averaged over the inputs, whereas `micro` contains the scores for each input.
    
## Implementation Notes
- We modified the script to run prediction because it did not save the scores of the model, just the overall labels.
The modified script can be found [here](src/run_test.py).
    
## Docker Information
- Image name: `kryscinski2019`
- Build command:
  ```shell script
  repro setup kryscinski2019 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup kryscinski2019
pytest models/kryscinski2019/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1099154015)
- [ ] Correctness unit tests pass  
No examples provided in the original repo
- [x] Model runs on full test dataset  
See our reproducibility experiment [here](experiments/reproduce-results/Readme.md)
- [x] Predictions approximately replicate results reported in the paper  
See our reproducibility experiment [here](experiments/reproduce-results/Readme.md)
- [ ] Predictions exactly replicate results reported in the paper  
Not tested
