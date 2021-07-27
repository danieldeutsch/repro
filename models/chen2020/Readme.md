# Chen et al. (2020)

## Publication
[MOCHA: A Dataset for Training and Evaluating Generative Reading Comprehension Metrics](https://arxiv.org/abs/2010.03636)

## Repositories
https://github.com/anthonywchen/MOCHA

## Available Models
This implementation contains a wrapper around the LERC model trained on all of the constituent datasets plus their evaluation script.

- [LERC](https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz)
  - Description: The LERC model trained on all datasets
  - Name: `chen2020-lerc`
  - Usage:
    ```python
    from repro.models.chen2020 import LERC
    model = LERC()
    score = model.predict("context", "question", "reference", "candidate")
    ```
    
- MOCHA Evaluation
  - Description: The MOCHA evaluation script that calculates the Pearson correlation between the ground-truth and predicted scores.
  - Name: `chen2020-eval`
  - Usage:
    ```python
    from repro.models.chen2020 import MOCHAEvaluationMetric
    model = MOCHAEvaluationMetric()
    # `inputs` should have the dataset, source, ground-truth score,
    # and predictions
    inputs = [
        {"dataset": dataset, "source": source, "score": score, "prediction": prediction},
        ...
    ]
    metrics = model.predict_batch(inputs)
    ```
    
## Implementation Notes
    
## Docker Information
- Image name: `chen2020`
- Build command:
  ```shell script
  repro setup chen2020 [--silent]
  ```
- Requires network: No
  
## Testing
Explain how to run the unittests for this model
```shell script
repro setup chen2020
pytest models/chen2020/tests
```

## Status
- [x] Regression unit tests pass   
See [here](https://github.com/danieldeutsch/repro/actions/runs/1071762824)
- [ ] Correctness unit tests pass  
No expected outputs provided in the original repo
- [x] Model runs on full test dataset  
See [here](experiments/reproduce-results/Readme.md)
- [x] Predictions approximately replicate results reported in the paper    
Yes, see [here](experiments/reproduce-results/Readme.md)
- [ ] Predictions exactly replicate results reported in the paper  
