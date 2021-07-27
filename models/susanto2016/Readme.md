# Susanto et al. (2016)

## Publication
[Learning to Capitalize with Character-Level Recurrent Neural Networks: An Empirical Study](https://aclanthology.org/D16-1225/)

## Repositories
The original repository for the paper is [this one](https://gitlab.com/raymondhs/char-rnn-truecase), but our implementation wraps a PyTorch version [here](https://github.com/mayhewsw/pytorch-truecaser) that is based loosely on the original paper.

## Available Models
The repository provides truecasing models for four different languages:

- [English](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0)
  - Description: An English model trained on Wikipedia
  - Name: `susanto2016-truecaser`
  - Usage:
    ```python
    from repro.models.susanto2016 import RNNTruecaser
    model = RNNTruecaser("wiki-truecaser-model-en.tar.gz")
    truecased = model.predict("text")
    ```
    
- [Spanish](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0)
  - Description: An Spanish model trained on WMT
  - Name: `susanto2016-truecaser`
  - Usage:
    ```python
    from repro.models.susanto2016 import RNNTruecaser
    model = RNNTruecaser("wmt-truecaser-model-es.tar.gz")
    truecased = model.predict("text")
    ```
    
- [German](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0)
  - Description: A German model trained on WMT
  - Name: `susanto2016-truecaser`
  - Usage:
    ```python
    from repro.models.susanto2016 import RNNTruecaser
    model = RNNTruecaser("wmt-truecaser-model-de.tar.gz")
    truecased = model.predict("text")
    ```
    
- [Russian](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0)
  - Description: A Russian model trained on LORELEI data
  - Name: `susanto2016-truecaser`
  - Usage:
    ```python
    from repro.models.susanto2016 import RNNTruecaser
    model = RNNTruecaser("lrl-truecaser-model-ru.tar.gz")
    truecased = model.predict("text")
    ```
    
## Implementation Notes
    
## Docker Information
- Image name: `susanto2016`
- Build command:
  ```shell script
  repro setup susanto2016 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup susanto2016
pytest models/susanto2016/tests
```

## Status
- [x] Regression unit tests pass  
See [here](https://github.com/danieldeutsch/repro/actions/runs/1058166921)
- [ ] Correctness unit tests pass  
No expected output provided
- [ ] Model runs on full test dataset    
Not tested
- [ ] Predictions approximately replicate results reported in the paper    
Not tested
- [ ] Predictions exactly replicate results reported in the paper    
Not tested