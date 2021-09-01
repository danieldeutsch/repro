# Kitaev et al. (2019)

## Publication
[Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/abs/1805.01052)

[Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760)

## Repositories
https://github.com/nikitakit/self-attentive-parser

## Available Models
- Benepar
  - Description: A wrapper around the Benepar parser
  - Name: `kitaev2019-benepar`
  - Usage:
    ```python
    from repro.models.kitaev2019 import Benepar
    model = Benepar()
    trees = model.predict("The time for action is now.")
    ```
    `trees` is a list of strings which contains the serialized parse trees for the input text.
    
## Implementation Notes
- The input text does not have to be a single sentence.
A parse tree will be returned for each one based on the library's sentence splitting logic.
    
## Docker Information
- Image name: `danieldeutsch/kitaev2019`
- Build command:
  ```shell script
  repro setup kitaev2019 [--models <model-name>+] [--silent]
  ```
  The `--models` argument specifies which pretrained parsing models should be included in the Docker image.
  By default, only the `benepar_en3` model is included.
  The list of available models can be found [here](https://github.com/nikitakit/self-attentive-parser#available-models).
- Requires network: No
  
## Testing
```shell script
repro setup kitaev2019
pytest models/kitaev2019/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
We verify the parser returns the example from the Github repo.
See [here](https://github.com/danieldeutsch/repro/actions/runs/1190222222).
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
Not tested
- [ ] Predictions exactly replicate results reported in the paper  
Not tested

## Changelog
