# FitzGerald et al. (2018)

## Publication
[Large-Scale QA-SRL Parsing](https://arxiv.org/abs/1805.05377)

## Repositories
https://github.com/nafitzgerald/nrl-qasrl

## Available Models
- QA-SRL Parser
  - Description: A QA-SRL parser
  - Name: `fitzgerald2018-qasrl-parser`
  - Usage:
    ```python
    from repro.models.fitzgerald2018 import QASRLParser
    model = QASRLParser()
    output = model.predict("The sentence to parse.")
    ```
    The `output` is a dictionary with the data output by the parser.
    See the original repository or our unit tests for an example.
    
## Implementation Notes

    
## Docker Information
- Image name: `danieldeutsch/fitzgerald2018`
- Build command:
  ```shell script
  repro setup fitzgerald2018
  ```
- Requires network: Yes, AllenNLP still sends a request even when a dataset is already cached locally.
  
## Testing
```shell script
repro setup fitzgerald2018
pytest models/fitzgerald2018/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1439953014)
- [x] Correctness unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1439953014).
The output is slightly different than what is expected in the Readme from the original repo, but it looks close enough.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
