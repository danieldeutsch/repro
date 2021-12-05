# Pyatkin et al. (2021)

## Publication
[Asking It All: Generating Contextualized Questions for any Semantic Role](https://arxiv.org/abs/2109.04832)

## Repositories
https://github.com/ValentinaPy/RoleQGeneration

This implementation uses our fork with some additional code modifications:
https://github.com/danieldeutsch/RoleQGeneration

## Available Models
- RoleQuestionGenerator
  - Description: A model which generates role questions.
  This model is the one released by the authors.
  - Name: `pyatkin2021-role-question-generator`
  - Usage:
    ```python
    from repro.models.pyatkin2021 import RoleQuestionGenerator
    model = RoleQuestionGenerator()
    inputs = [
        {
            "sentence": "Tom brings the dog to the park.",
            "token_index": 1,
            "lemma": "bring",
            "pos": "v",
            "sense": 1
        }
    ]
    outputs = model.predict_batch(inputs)
    ```
    The input keys are the sentence, the token index of the predicate in the sentence, the lemma of the predicate in OntoNotes, the part-of-speech of the predicate, and the OntoNotes sense index of the predicate. 
    
## Implementation Notes
    
## Docker Information
- Image name: `danieldeutsch/pyatkin2021:1.0`
- Build command:
  ```shell script
  repro setup pyatkin2021
  ```
- Requires network: Yes, the code does a network call related to ensuring NLTK libraries are installed.
  
## Testing
```shell script
repro setup pyatkin2021
pytest models/pyatkin2021/tests
```

## Status
- [x] Regression unit tests pass  
- [ ] Correctness unit tests pass  
- [ ] Model runs on full test dataset  
- [ ] Predictions approximately replicate results reported in the paper  
- [ ] Predictions exactly replicate results reported in the paper  

## Changelog
