# Scialom et al. (2021)

## Publication
[QuestEval: Summarization Asks for Fact-based Evaluation](https://arxiv.org/abs/2103.12693)

## Repositories
https://github.com/ThomasScialom/QuestEval

## Available Models
This implementation wraps the `QuestEval` metric.

- QuestEval
  - Description: A QA-based metric that uses an optional source document and/or reference 
  - Name: `scialom2021-questeval`
  - Usage:
    ```python
    from repro.models.scialom2021 import QuestEval
    model = QuestEval()
    # Either "references" and/or "sources" must be present. Only supports single
    # references/documents
    inputs = [
        {"candidate": "The candidate", "references": ["The reference"], "sources": ["The source"]},
        ...
    ]   
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` is the averaged QuestEval scores over the inputs, and `micro` is the individual scores per input.
    Any `**kwargs` passed to `predict` or `predict_batch` will be passed to the constructor of the QuestEval metric in the original code.
    
- QuestEvalForSummarization
  - Description: A wrapper around `QuestEval` which passes the arguments specific to summarization to the original code's QuestEval constructor by default.
  - Name: `scialom2021-questeval-summarization`
  - Usage: 
    ```python
    from repro.models.scialom2021 import QuestEvalForSummarization
    model = QuestEvalForSummarization()
    # Either "references" and/or "sources" must be present. Only supports single
    # references/documents
    inputs = [
        {"candidate": "The candidate", "references": ["The reference"], "sources": ["The source"]},
        ...
    ]   
    macro, micro = model.predict_batch(inputs)
    ```
    
- QuestEvalForSimplification
  - Description: A wrapper around `QuestEval` which passes the arguments specific to simplification to the original code's QuestEval constructor by default.
  - Name: `scialom2021-questeval-simplification`
  - Usage: 
    ```python
    from repro.models.scialom2021 import QuestEvalForSimplification
    model = QuestEvalForSimplification()
    # Either "references" and/or "sources" must be present. Only supports single
    # references/documents
    inputs = [
        {"candidate": "The candidate", "references": ["The reference"], "sources": ["The source"]},
        ...
    ]   
    macro, micro = model.predict_batch(inputs)
    ```
    
## Implementation Notes
- This implementation is based on the `v0.0.1` tag of the QuestEval repro based on the authors' recommendation for the correct API for the summarization task.
    
## Docker Information
- Image name: `scialom2021`
- Build command:
  ```shell script
  repro setup scialom2021 [--silent]
  ```
- Requires network: No
  
## Testing
Setting `TEST_DEVICES=<gpu-id>` is required for the tests.
```shell script
repro setup scialom2021
pytest models/scialom2021/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
We have reproduced the examples from the original code's Github repo.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested