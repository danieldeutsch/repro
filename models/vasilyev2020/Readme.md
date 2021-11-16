# Vasilyev et al. (2020)

## Publication
[Fill in the BLANC: Human-free quality estimation of document summaries](https://aclanthology.org/2020.eval4nlp-1.2/)

## Repositories
https://github.com/PrimerAI/blanc

## Available Models
- BLANCHelp
  - Description: A reference-free summarization evaluation metric
  - Name: `vasilyev2020-blanc-help`
  - Usage:
    ```python
    from repro.models.vasilyev2020 import BLANCHelp
    model = BLANCHelp()
    inputs = [
        {"sources": ["The input document"], "candidate": "The summary"}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The output `micro` is the input-level score.
    `macro` is the score averaged over all of the inputs.
    
- BLANCTune
  - Description: A reference-free summarization evaluation metric
  - Name: `vasilyev2020-blanc-tune`
  - Usage:
    ```python
    from repro.models.vasilyev2020 import BLANCTune
    model = BLANCTune()
    inputs = [
        {"sources": ["The input document"], "candidate": "The summary"}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The output `micro` is the input-level score.
    `macro` is the score averaged over all of the inputs.
    
## Implementation Notes
The unit tests here do not replicate the example outputs from the original repository.
However, there were also differences between the original Readme and [this Colab notebook](https://colab.research.google.com/drive/17pJ94L2kCL6QMBMflOm-H0ApBiOUWJ1H?usp=sharing) which reruns the provided examples.
Further, the scores (at least for `BLANCTune`) are different whether you use CPU or GPU.
    
## Docker Information
- Image name: `danieldeutsch/vasilyev2020`
- Build command: Provide documentation on how to build the image
  ```shell script
  repro setup vasilyev2020
  ```
- Requires network: Yes, it still requires connecting to the network even when warmup queries are run.
  
## Testing
```shell script
repro setup vasilyev2020
pytest models/vasilyev2020/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1467631310).
- [ ] Correctness unit tests pass  
See explanation in "Implementation Notes"
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
