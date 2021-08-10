# Thompson & Post (2020)

## Publication
[Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing](https://arxiv.org/abs/2004.14564)

## Repositories
https://github.com/thompsonb/prism

## Available Models
- Prism
  - Description: A machine translation metric based on paraphrasing
  - Name: `thompson2020-prism`
  - Usage:
    ```python
    from repro.models.thompson2020 import Prism
    model = Prism()
    inputs = [
        {"candidate": "The candidate", "references": ["The reference"]}
    ]
    macro, micro = model.predict(inputs)
    
    inputs = [
        {"candidate": "The candidate", "sources": ["The source"]}
    ]
    macro, micro = model.predict(inputs)
    ```
    `macro` is the average Prism score across inputs, and `micro` is the score per input.
    
## Implementation Notes
- The metric requires all inputs to have sources xor references, not both or neither.
- The metric only supports single references and/or sources, so the length of `references` and `sources` must be 1.

## Docker Information
- Image name: `danieldeutsch/thompson2020:1.0`
- Build command:
  ```shell script
  repro setup thompson2020 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup thompson2020
pytest models/thompson2020/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1117854308).
We replicate the output provided in their Github repository.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
