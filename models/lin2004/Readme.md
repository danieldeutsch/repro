# Lin (2004)

## Publication
[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)

## Repositories
n/a

## Available Models
- ROUGE
  - Description: A wrapper around the original Perl implementation of ROUGE
  - Name: `lin2004-rouge`
  - Usage:
    ```python
    from repro.models.lin2004 import ROUGE
    model = ROUGE()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` results are the ROUGE scores averaged over the `inputs`.
    The `micro` results are the ROUGE results for each item in `inputs`.
    
## Implementation Notes
    
## Docker Information
- Image name: `lin2004`
- Docker Hub: https://hub.docker.com/repository/docker/danieldeutsch/lin2004
- Build command:
  ```shell script
  repro setup lin2004 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup lin2004
pytest models/lin2004/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1083114293)
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
n/a  
- [ ] Predictions exactly replicate results reported in the paper  
n/a