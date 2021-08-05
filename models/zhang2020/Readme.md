# Zhang et al. (2020)

## Publication
[BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)

## Repositories
https://github.com/Tiiiger/bert_score

## Available Models
This implementation wraps BERTScore.
Each of the backend models can be accessed by passing the name as an argument to the class constructor.
Models can be pre-cached by passing its name to the `setup` command (see below).
By default, the default English model is used unless otherwise specified.

- BERTScore
  - Description: An text generation evaluation metric based on BERT.
  - Name: `zhang2020-bertscore`
  - Usage:
    ```python
    from repro.models.zhang2020 import BERTScore
    model = BERTScore()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` results are the BERTScore scores averaged over the `inputs`.
    The `micro` results are the BERTScore results for each item in `inputs`.
    
## Implementation Notes
- If you are going to repeatedly use a model, it is better to run the `setup` command and pass that model's name.
Otherwise, every time you run the metric, the model will be downloaded again.

- This implementation will return a score of 0 for precision, recall, and F1 if the input is empty.
This is not true of the original code, which returns a non-zero recall when this is true.
    
## Docker Information
- Image name: `zhang2020`
- Build command:
  ```shell script
  repro setup zhang2020 \
    [--models <model-name>+] \
    [--silent]
  ```
  The `--models` argument specifies which BERTScore backend models should be cached in the Docker image.
  By default, this includes `roberta-large`, the default model for English. 
  
- Requires network: Yes, it still tries to request for a file from the web even if it is cached locally.
  
## Testing
```shell script
repro setup zhang2020
pytest models/zhang2020/tests
```

## Status
- [x] Regression unit tests pass  
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1102240656).
The tests check for the same scores as in the original repo's unit tests.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested