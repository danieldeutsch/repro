# Goyal & Durrett (2020)

## Publication
[Evaluating Factuality in Generation with Dependency-level Entailment](https://aclanthology.org/2020.findings-emnlp.322/)

## Repositories
https://github.com/tagoyal/dae-factuality

## Available Models
This implementation wraps the DAE evaluation metric.
There are three versions available, `dae_basic`, `dae_w_syn` and `dae_w_syn_hallu`, which can be configured using the `model` parameter to the constructor.

- DAE
  - Description: A dependency-based factuality metric
  - Name: `goyal2020-dae`
  - Usage:
    ```python
    from repro.models.goyal2020 import DAE
    # "dae_w_syn" is the default model
    model = DAE()
    inputs = [
        {"candidate": "The candidate sentence", "sources": ["The source sentence"]}
    ]
    maco, micro = model.predict_batch(inputs)
    ```
    `macro` is the average DAE score over the inputs, and `micro` is the individual DAE scores per input.
    
## Implementation Notes
- The implementation only allows for a single source, so the length of `"sources"` must be 1.
    
## Docker Information
- Image name: `goyal2020`
- Build command:
  ```shell script
  repro setup goyal2020 [--silent]
  ```
- Requires network: Yes, the CoreNLP server uses the network.
  
## Testing
```shell script
repro setup goyal2020
pytest models/goyal2020/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1105599382).
The regression tests for "dae_basic" and "dae_w_syn" are not very strong since the scores are all around 0.9999.
- [ ] Correctness unit tests pass  
No example outputs provided by the original repo.
- [X] Model runs on full test dataset  
See [here](experiments/reproduce-results/Readme.md)
- [x] Predictions approximately replicate results reported in the paper  
- [x] Predictions exactly replicate results reported in the paper  
See [here](experiments/reproduce-results/Readme.md)