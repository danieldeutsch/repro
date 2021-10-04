# Zhang & Bansal (2021)

## Publication
[Finding a Balanced Degree of Automation for Summary Evaluation](https://arxiv.org/abs/2109.11503)

## Repositories
https://github.com/ZhangShiyue/Lite2-3Pyramid

## Available Models
- Lite3Pyramid
  - Description: An automated Pyramid Score based on SRL
  - Name: `zhang2021-lite3pyramid`
  - Usage:
    ```python
    from repro.models.zhang2021 import Lite3Pyramid
    model = Lite3Pyramid()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The references"]}
    ]
    macro, micro = model.predict(inputs)
    
    inputs = [
        {"candidate": "The candidate summary", "units_list": [["STU 1 for reference 1", "STU 2"]]}
    ]
    macro, micro = model.predict(inputs)
    ```
    `macro` is the Lite3Pyramid scores averaged over the inputs.
    `micro` is the per-input scores, each averaged over the references per input. 
    
## Implementation Notes

    
## Docker Information
- Image name: `danieldeutsch/zhang2021:1.0`
- Docker Hub: 
- Build command:
  ```shell script
  repro setup zhang2021 [--models <model-name>+]
  ```
  The `--models` argument specifics which pretrained NLI models will be pre-cached inside of the Docker image.
  See [here](https://github.com/ZhangShiyue/Lite2-3Pyramid) for the available models.
  
- Requires network: Yes, AllenNLP sends a request for a model even if the model is available locally.
  
## Testing
```shell script
repro setup zhang2021
pytest models/zhang2021/tests
```
Most of the tests require using a GPU for speed purposes.

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
The STU extraction gives slightly different results, but calculating the scores given a summary and STUs gives the expected result.

- [ ] Model runs on full test dataset  
Not tested
- [x] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Changelog
