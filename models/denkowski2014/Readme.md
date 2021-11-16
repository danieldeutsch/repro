# Denkowski & Lavie (2014)

## Publication
[Meteor Universal: Language Specific Translation Evaluation for Any Target Language](https://aclanthology.org/W14-3348/)

## Repositories
https://www.cs.cmu.edu/~alavie/METEOR/index.html#Download

## Available Models
- METEOR (v1.5)
  - Description: An alignment-based translation evaluation metric
  - Name: `denkowski2014-meteor`
  - Usage:
    ```python
    from repro.models.denkowski2014 import METEOR
    model = METEOR()
    inputs = [
        {"candidate": "The candidate text", "references": ["The first", "The second reference"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` and `micro` are the averaged and input-level METEOR scores.
    
## Implementation Notes
    
## Docker Information
- Image name: `danieldeutsch/denkowski2014`
- Build command:
  ```shell script
  repro setup denkowski2014
  ```
- Requires network: No
  
## Testing
```shell script
repro setup denkowski2014
pytest models/denkowski2014/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1468357287)
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Changelog
