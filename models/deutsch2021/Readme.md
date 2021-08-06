# Deutsch et al. (2021)

## Publication
[Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary](https://arxiv.org/abs/2010.00490)

## Repositories
https://github.com/danieldeutsch/qaeval

## Available Models
We have implemented the QAEval metric as well as its question-generation and question-answering models.

- QAEval:
  - Description: A question-answering reference-based summarization evaluation metric.
  - Name: `deutsch2021-qaeval`
  - Usage:
    ```python
    from repro.models.deutsch2021 import QAEval
    model = QAEval()
    inputs = [
        {"candidate": "The candidate summary", "references": ["The first reference", "The second"]}
    ]
    macro, micro = model.predict_batch(inputs)
    ```
    The `macro` results are the QAEval scores averaged over the `inputs`.
    The `micro` results are the QAEval results for each item in `inputs`.
    
    You can also return the QA pairs for each input with the `return_qa_pairs=True` flag:
    ```python
    macro, micro, qa_pairs = model.predict_batch(inputs, return_qa_pairs=True)
    ```

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
- Docker Hub: https://hub.docker.com/repository/docker/danieldeutsch/deutsch2021
- Build command:
  ```shell script
  repro setup deutsch2021 [--silent]
  ```
- Requires network: No
  
## Testing
```shell script
repro setup deutsch2021
pytest models/deutsch2021/tests
```

## Status
- [x] Regression unit tests pass   
- [x] Correctness unit tests pass  
The tests were taken from the `qaeval` and `sacrerouge` repositories.
See [here](https://github.com/danieldeutsch/repro/actions/runs/1063451340) (these tests don't include the QAEval metric test, which likely takes too long on the CPU).
- [ ] Model runs on full test dataset  
Not tested
- [x] Predictions approximately replicate results reported in the paper  
The question-answering model replicates the expected results (see [here](experiments/reproduce-results/Readme.md)).
The question-generation model was not quantitatively evaluated in the paper.
We did not test QAEval on the full dataset, but the scores match the examples from the `qaeval` and `sacrerouge` repos.
- [ ] Predictions exactly replicate results reported in the paper  
Not tested