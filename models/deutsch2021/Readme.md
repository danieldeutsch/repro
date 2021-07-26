# Deutsch et al. (2021)

## Publication
[Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary](https://arxiv.org/abs/2010.00490)

## Repositories
https://github.com/danieldeutsch/qaeval

## Available Models
We have implemented the question-generation and question-answering models from the QAEval metric.

- [Question Generation](https://drive.google.com/file/d/1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU/view)
  - Description: A question-generation model
  - Name: `deutsch2021-question-generation`
  - Usage:
    ```python
    from repro.models.deutsch2021 import QAEvalQuestionGenerationModel
    model = QAEvalQuestionGenerationModel()
    context = "My name is Dan."
    start, end = 11, 14  # "Dan", end is exclusive
    question = model.predict(context, start, end)
    ```
    
- [Question Answering](https://drive.google.com/file/d/1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8/view)
  - Description: A question-answering model trained on SQuAD 2.0
  - Name: `deutsch2021-question-answering`
  - Usage:
    ```python
    from repro.models.deutsch2021 import QAEvalQuestionAnsweringModel
    model = QAEvalQuestionAnsweringModel()
    context = "My name is Dan."
    question = "What is my name?"
    # If the answer is `None`, that means the model predicted that
    # the question is not answerable
    answer = model.predict(context, question)
    # Add `return_dicts=True` to get more metadata about the answer.
    # The "prediction" key will be the model's best non-null choice
    answer_dict = model.predict(context, question, return_dicts=True)
    ```
    
## Implementation Notes
    
## Docker Information
- Image name: `deutsch2021`
- Build command:
  ```shell script
  repro setup deutsch2021 \
    [--question-generation] \
    [--question-answering] \
    [--silent]
  ```
  The arguments specify which pretrained models should be downloaded
- Requires network: No
  
## Testing
Explain how to run the unittests for this model
```shell script
repro setup deutsch2021 \
    --question-generation \
    --question-answering
pytest models/deutsch2021/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
The tests were taken from the `qaeval` repository. See [here](https://github.com/danieldeutsch/repro/actions/runs/1063451340).
- [ ] Model runs on full test dataset  
Not tested
- [ ] Predictions approximately replicate results reported in the paper  
The question-answering model replicates the expected results (see [here](experiments/reproduce-results/Readme.md)).
The question-generation model was not quantitatively evaluated in the paper.
- [ ] Predictions exactly replicate results reported in the paper  
Not tested