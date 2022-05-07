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
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` is the average Prism score across inputs, and `micro` is the score per input.
    
- Prism-src
    - Description: A reference-free machine translation metric based on paraphrasing
  - Name: `thompson2020-prism-src`
  - Usage:
    ```python
    from repro.models.thompson2020 import PrismSrc
    model = PrismSrc()
    inputs = [
        {"candidate": "The candidate", "sources": ["The source"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    `macro` is the average Prism-src score across inputs, and `micro` is the score per input.
    
- Prism as an MT model
  - Description: Uses Prism as a machine translation model
  - Name: `thompson2020-prism`
  - Usage:
    ```python
    from repro.models.thompson2020 import Prism
    model = Prism()
    inputs = [
        {"source": "The source text."}
    ]
    outputs = model.translate_batch("fr", inputs)
    ```
    `"fr"` is the language code for the target language and `outputs` is the list of translations.

## Implementation Notes
- The metric requires all inputs to have sources xor references, not both or neither.
- The metric only supports single references and/or sources, so the length of `references` and `sources` must be 1.

## Docker Information
- Image name: `danieldeutsch/thompson2020:1.2`
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
See [here](https://github.com/danieldeutsch/repro/actions/runs/2287250116).
We replicate the output provided in their Github repository.
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
### v1.3
- Separated Prism and PrismSrc.
The output metric name for PrismSrc is now "prism-src"

### v1.2
- Added a `batch_size` argument to the translation functions

### v1.1
- Added using Prism as an MT model
