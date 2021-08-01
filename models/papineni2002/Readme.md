# Papineni et al. (2002)

## Publication
[BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

## Repositories
https://github.com/mjpost/sacrebleu

## Available Models
Our implementation wraps BLEU and the sentence-level version, SentBLEU.

- BLEU
  - Description: The BLEU metric
  - Name: `papineni2002-bleu`
  - Usage:
    ```python
    from repro.models.papineni2002 import BLEU
    model = BLEU()
    inputs = [
        {"candidate": "The candidate", "references": ["Reference one", "The second"]},
        ...
    ]
    macro, _ = model.predict_batch(inputs)
    ```
    `macro` will contain the BLEU score.
    Since BLEU is a corpus-level metric, there is not input-level score (thus we ignore the second returned value, which is an empty list)
    
- SentBLEU
  - Description: The sentence-level BLEU metric
  - Name: `papineni2002-sentbleu`
  - Usage:
    ```python
    from repro.models.papineni2002 import SentBLEU
    model = SentBLEU()
    inputs = [
        {"candidate": "The candidate", "references": ["Reference one", "The second"]},
        ...
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` contains the average SentBLEU score over the inputs.
    `micro` contains the SentBLEU score for each input.
    
## Implementation Notes
    
## Docker Information
- Image name: `papineni2002`
- Build command:
  ```shell script
  repro setup papineni2002 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup papineni2002
pytest models/papineni2002/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1087885929).
We check a corpus-level example from the SacreBLEU unit tests.
We did not test for SentBLEU's correctness or BLEU with a different number of references per hypothesis.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
n/a
- [ ] Predictions exactly replicate results reported in the paper  
n/a